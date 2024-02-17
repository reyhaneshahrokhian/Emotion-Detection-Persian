import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel, BertConfig, BertTokenizer
from sa_project import test , class_finder , one_hot_to_string , MultiLabelDataset , tokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

def eval(path):
    MAX_LEN = 256
    df = pd.read_csv(path, sep='\t', names=['text', 'labels'], encoding='latin1')
    test_rnd_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                    }

    test_rnd_set = MultiLabelDataset(df, tokenizer, MAX_LEN)
    testing_rnd_loader = DataLoader(test_rnd_set, **test_rnd_params)

    outputs, targets = test(testing_rnd_loader)
    final_outputs_rnd = class_finder(outputs)

    for ind in df.index:
        print(f"The text is :{df['text'][ind]}")
        print(f"The True Label is : {one_hot_to_string(df['labels'][ind])}")
        print(f"The Predicted label is {one_hot_to_string(list(final_outputs_rnd[ind]))}")
