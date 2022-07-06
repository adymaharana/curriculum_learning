## On Curriculum Learning for Commonsense Reasoning

PyTorch code for the NAACL 2022 paper "On Curriculum Learning for Commonsense Reasoning"


#### Requirements:
This code has been tested on torch==1.9.0 and transformers==4.3.2. Other required packages are [bayes_opt](https://pypi.org/project/bayesian-optimization/) and [tqdm](https://pypi.org/project/tqdm/).

#### Prepare Repository:
Download the datasets used in the paper from the following locations:
[SocialIQA](https://leaderboard.allenai.org/socialiqa/submissions/get-started)
[CosmosQA](https://leaderboard.allenai.org/cosmosqa/submissions/public)
[CODAH](https://github.com/yangyiben/G-DAUG-c-Generative-Data-Augmentation-for-Commonsense-Reasoning/tree/master/CODAH)
[ProtoQA](https://leaderboard.allenai.org/protoqa/submissions/get-started)
[WinoGrande](https://leaderboard.allenai.org/winogrande/submissions/get-started)
[GLUE](https://gluebenchmark.com/)

Save to the ```./data/``` folder. Our split of the HellaSWAG-2K dataset is available at the ```./data/hellaswag/``` folder.

#### Training RoBERTa Models for Results in Table 1 in paper:

The training is performed in two stages. Example commands are shown for the CosmosQA dataset. Scripts for other datasets are available in the repository.
- Stage 1: In this stage, the teacher models are trained via finetuning of pretrained RoBERTa-Large models using the standard method of randomly sampled training mini-batches. This also serves as the baseline RoBERTa model in our experiments. We provide scripts to find the best hyperparameters for the teacher model using bayesian optimization. To find the best RoBERTa model for CosmosQA, run\
```
cd roberta
python grid_search_hyperparams_roberta_cosmosqa.py
```
The default location for the model checkpoints is ```./baselines/```. For SocialIQA, CODAH and WinoGrande, we use the best hyperparameters reported in existing literature.

- Ranking of training samples: In this preprocessing step before Stage 2, the predictions from teacher models are used to rank the training samples in the order of difficulty. To get predictions for CosmosQA, run
```
bash cosmos.sh eval_valid
python process_logits.py
```

- Stage 3: In this stage, the student models are trained via curriculum learning using the ranked dataset from the previous step. We use bayesian optimization to find the best parameters for the pacing function; scripts for performing the optimization are available for all datasets in the repository. To find the best pacing function for CosmosQA, run:
``` 
python search_pacing_function_params_cosmosqa.py
```


## Citation:
```
@inproceedings{maharana2022oncurriculum,
  title={On Curriculum Learning for Commonsense Reasoning},
  author={Maharana, Adyasha and Bansal, Mohit},
  booktitle={NAACL},
  year={2022}
}
```