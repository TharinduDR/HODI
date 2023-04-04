import os
import argparse
import shutil
import pandas as pd
import torch
import sklearn
import statistics
from experiments.text_classification.evaluation import macro_f1, weighted_f1, sentence_label_evaluation
from experiments.text_classification.transformer_config import TEMP_DIRECTORY, SEED, transformer_args
from sklearn.model_selection import train_test_split

from text_classification.text_classification_model import TextClassificationModel

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)

arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

data = pd.read_csv("data/HODI_2023_train_subtaskA.tsv", sep="\t")
data = data.rename(columns={'homotransphobic': 'labels'})
data = data[['text', 'labels']]

macro_f1_scores = []
weighted_f1_scores = []

for i in range(transformer_args["n_fold"]):
    if os.path.exists(transformer_args['output_dir']) and os.path.isdir(transformer_args['output_dir']):
        shutil.rmtree(transformer_args['output_dir'])

    full_train, test = train_test_split(data, test_size=0.2, random_state=SEED*i)
    test_sentences = test['text'].tolist()

    print("Started Fold {}".format(i))
    model = TextClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_args,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(full_train, test_size=0.1, random_state=SEED * i)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    macro_f1, weighted_f1 = sentence_label_evaluation(predictions, test['labels'].tolist())
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)

print("Weighted F1 scores ", weighted_f1_scores)
print("Mean weighted F1 scores", statistics.mean(weighted_f1_scores))
print("STD weighted F1 scores", statistics.stdev(weighted_f1_scores))

print("Macro F1 scores ", macro_f1_scores)
print("Mean macro F1 scores", statistics.mean(macro_f1_scores))
print("STD macro F1 scores", statistics.stdev(macro_f1_scores))




