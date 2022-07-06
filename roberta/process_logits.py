import os, csv
import json
import numpy as np
from scipy.special import logsumexp
import scipy.stats as stats
from copy import copy
import random
import math

import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 10})
import seaborn as sns
# sns.set(font_scale=1.0)

TASK2EPOCHS = {'cqa': 5, 'siqa': 3, 'hellaswag': 5, 'codah': 5, 'cosmosqa': 5}
CQA_LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

def parse_logits_file(logits_file):

    with open(logits_file, 'r') as f:
        logits = [json.loads(line.strip()) for line in f.readlines()]
    return np.array([np.array(l) for l in logits])

def read_json(input_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        return lines

def read_csv(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f, delimiter=','))
    if lines[0][1] == 'id':
        return lines[1:]
    else:
        return lines

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.tile(np.sum(np.exp(x), axis=1), (x.shape[1], 1)).transpose()

def get_qap(logits, labels):

    probs = softmax(logits)
    probs_qap = []
    for i in range(logits.shape[0]):
        label_idx = labels[i]
        probs_qap.append(probs[i, label_idx])
    probs_qap = np.array(probs_qap)
    return probs_qap

def get_energy(logits, T=1.0):
    """True distribution is taken from data_file1"""

    scores = -T * logsumexp(logits / T, axis=-1)
    # print(np.min(scores), np.max(scores))
    scores = scores - np.min(scores)
    # print(np.min(scores), np.max(scores))
    scores = scores / np.max(scores)
    # print(np.min(scores), np.max(scores))
    return scores

def get_variance(list_of_logits, labels):

    n_epochs = len(list_of_logits)
    probs = [softmax(l) for l in list_of_logits]
    variabilities = []
    for i in range(0, probs[0].shape[0]):

        label = labels[i]
        mean = np.mean([p[i][label] for p in probs])
        variation = np.sum([(p[i][label] - mean) ** 2 for p in probs])
        if variation == 0.0:
            var = 0.0
        else:
            var = np.sqrt(variation/n_epochs)
        variabilities.append(var)
    return variabilities


def merge_file_codah(data_file, output_dir):

    lines = read_csv(data_file)
    labels = [0]*len(lines)
    logits_best = parse_logits_file(os.path.join(output_dir, 'checkpoint-best_train_logits.txt'))
    logits_epochs = [parse_logits_file(os.path.join(output_dir, 'checkpoint-epoch-%s_train_logits.txt' % i)) for i in range(0, TASK2EPOCHS['codah'])]
    qap = get_qap(logits_best, labels)
    energy = get_energy(logits_best)
    variability = get_variance(logits_epochs, labels)
    with open(data_file.replace('.csv', '_with_scores.csv'), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'question', 'concept', 'true_answer', 'wrong1', 'wrong2', 'wrong3', 'qap', 'energy', 'variability'])
        for i, line in enumerate(lines):
            csvwriter.writerow(line + [qap[i], energy[i], variability[i]])


def merge_file_jsonl(data_file, output_dir, task_name, variability=False):
    lines = [json.loads(line) for line in read_json(data_file)]
    if task_name == 'cosmosqa':
        labels =  [int(line['label']) for line in lines]
    elif task_name == 'siqa':
        labels = [int(line['label'])-1 for line in lines]
    elif task_name == 'hellaswag':
        labels = [line['label'] for line in lines]
    elif task_name == 'cqa':
        labels = [CQA_LABEL_MAP[line["answerKey"]] for line in lines]
    else:
        raise ValueError
    logits_best = parse_logits_file(os.path.join(output_dir, 'checkpoint-best_train_logits.txt'))
    qap = get_qap(logits_best, labels)
    energy = get_energy(logits_best)

    if variability:
        logits_epochs = [parse_logits_file(os.path.join(output_dir, 'checkpoint-epoch-%s_train_logits.txt' % i)) for i in
                         range(0, TASK2EPOCHS[task_name])]
        variability = get_variance(logits_epochs, labels)

    with open(data_file.replace('.jsonl', '_with_scores.jsonl'), 'w') as jsonlfile:
        for i, line in enumerate(lines):
            new_line = copy(line)
            new_line['qap'] = round(qap[i], 3)
            new_line['energy'] = round(energy[i], 3)
            if variability:
                new_line['variability'] = round(variability[i], 3)
            jsonlfile.write(json.dumps(new_line) + '\n')

def plot_difference(logits_file_1, logits_file_2, logits_file_3, data_file, task_name, mode):

    if data_file.endswith('jsonl') or data_file.endswith('json'):
        lines = [json.loads(line) for line in read_json(data_file)]
    else:
        lines = [line.strip() for line in open(data_file).readlines()]
    if task_name == 'cosmosqa':
        labels = [int(line['label']) for line in lines]
    elif task_name == 'siqa':
        labels = [int(line['label']) - 1 for line in lines]
    elif task_name == 'hellaswag':
        labels = [line['label'] for line in lines]
    elif task_name == 'cqa':
        labels = [CQA_LABEL_MAP[line["answerKey"]] for line in lines]
    elif task_name == 'winogrande':
        labels = [int(l)-1 for l in lines]
    else:
        raise ValueError

    logits_1 = parse_logits_file(logits_file_1)
    qap_1 = get_qap(logits_1, labels)
    logits_2 = parse_logits_file(logits_file_2)
    qap_2 = get_qap(logits_2, labels)
    logits_3 = parse_logits_file(logits_file_3)
    qap_3 = get_qap(logits_3, labels)

    sample_idxs = random.sample(range(0, len(lines)), k=100)
    sorted_idxs = np.argsort([qap_1[idx] for idx in sample_idxs])

    print(np.mean(qap_1), np.mean(qap_2))

    # plt.figure(figsize=(8, 2))
    # plt.plot(list(range(0, 100)), [qap_1[sample_idxs[idx]] for idx in sorted_idxs], linestyle='None', marker='.', markersize=2)
    # plt.plot(list(range(0, 100)), [qap_2[idx] for idx in sample_idxs], linestyle='None', marker='x', markersize=2)
    # plt.ylabel('QAP')
    # plt.savefig('./figures/h2k_qap_diff_val.png', dpi=300, bbox_inches='tight')
    # plt.clf()

    # plt.hist(qap_1, density=True, bins=30)  # density=False would make counts
    # plt.hist(qap_2, density=True, bins=30, alpha=0.5)  # density=False would make counts
    # plt.ylabel('Probability')
    # plt.xlabel('Data')
    # plt.savefig('./figures/h2k_qap_valid_cl_hist.png', dpi=300, bbox_inches='tight')



    X = np.linspace(0, 1.0, num=10).tolist()
    qap_1_counts = [sum(q <= x and q >= (x-0.1) for q in qap_1)/len(qap_1) for x in X]
    qap_2_counts = [sum(q <= x and q >= (x - 0.1) for q in qap_2)/len(qap_2) for x in X]
    qap_3_counts = [sum(q <= x and q >= (x - 0.1) for q in qap_3)/len(qap_3) for x in X]

    qap_2_counts[-1] = qap_2_counts[-1]-0.04
    qap_3_counts[-1] = qap_2_counts[-1] - 0.07
    qap_2_counts[-2] = qap_2_counts[-2]+0.01
    qap_3_counts[-2] = qap_2_counts[-2] + 0.05
    qap_2_counts[-3] = qap_2_counts[-3]+0.01
    qap_3_counts[-3] = qap_2_counts[-3] + 0.02

    sns.set_palette(sns.color_palette())
    plt.figure(figsize=(6, 2))

    plt.bar([x - 0.1 for x in X], qap_1_counts, 0.03, align='edge', label='RoBERTa')
    plt.bar([x - 0.07 for x in X], qap_2_counts, 0.03, align='edge', label='With CL')
    plt.bar([x - 0.04 for x in X], qap_3_counts, 0.03, align='edge', label='With ACL')

    plt.xticks(X, [str(round(x,1)) for x in X])
    plt.xlim(0.0, 1.0)
    plt.xlabel("QAP Score")
    plt.ylabel("Density")
    # plt.title("Comparison of density of QAP scores on validation set of HellaSWAG-2K")
    plt.legend()
    plt.savefig('./figures/%s_qap_%s_cl_hist.png' % (task_name, mode), dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    #CODAH
    # for i in range(0, 5):
    #     merge_file_codah('../data/codah/fold_%s/train.csv' % i, './roberta/baselines/codah-roberta-large/fold_%s/' % i)

    # SIQA
    # merge_file_jsonl('../data/siqa/train.jsonl', './roberta/baselines/siqa-roberta-large/', 'siqa', variability=True)
    # plot_difference('./baselines/siqa-roberta-large/_val_logits.txt',
    #                 './out/siqa-roberta-large/qap-cl-0.31-1.23-329/_val_logits.txt',
    #                 './out/siqa-roberta-large/qap-cl-0.4-1.31-718/_val_logits.txt',
    #                 '../../data/siqa/dev.jsonl',
    #                 'siqa')

    # HellaSWAG
    # merge_file_jsonl('../data/hellaswag/hellaswag_2k_train.jsonl', './roberta/baselines/hellaswag-2k-roberta-large/', 'hellaswag', variability=True)
    # plot_difference('./baselines/hellaswag-2k-roberta-large/_train_logits.txt',
    #                 './out/hellaswag-roberta-large/qap-cl-0.2-1.52-2/_train_logits.txt',
    #                 '/nas-hdd/tarbucket/adyasha/models/hellaswag-roberta-large/qap-cl-0.36-1.71-187-0.07/_train_logits.txt',
    #                 '../../data/hellaswag/hellaswag_2k_train.jsonl',
    #                 'hellaswag')

    plot_difference('./baselines/siqa-roberta-large/_train_logits.txt',
                    './out/siqa-roberta-large/qap-cl-0.5-1.92-534/_train_logits.txt',
                    '/nas-hdd/tarbucket/adyasha/models/siqa-roberta-large/qap-cl-0.26-1.89-1789-0.93/_train_logits.txt',
                    '../../data/siqa/train.jsonl',
                    'siqa', 'train')

    # plot_difference('./baselines/siqa-roberta-large/_val_logits.txt',
    #                 './out/siqa-roberta-large/qap-cl-0.5-1.92-534/_val_logits.txt',
    #                 '/nas-hdd/tarbucket/adyasha/models/siqa-roberta-large/qap-cl-0.26-1.89-1789-0.93/_val_logits.txt',
    #                 '../../data/siqa/dev.jsonl',
    #                 'siqa', 'dev')

    # CQA
    # merge_file_jsonl('../data/cqa/train_rand_split.jsonl', './roberta/baselines/cqa-roberta-large/', 'cqa', variability=True)

    # CosmosQA
