# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb

from torch import cuda
import logging

logger = logging.getLogger("mydataset")
logging.basicConfig(level=logging.DEBUG)

class MyDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, target_len, is_eval=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.question = self.data.question
        self.ans = self.data.ans
        self.is_eval = is_eval

    def __len__(self):
        return len(self.question)

    def __getitem__(self, index):
        text = str(self.question[index])
        text = ' '.join(text.split())

        ctext = str(self.ans[index])
        ctext = ' '.join(ctext.split())
        source = self.tokenizer.batch_encode_plus([text], pad_to_max_length=True, max_length=self.source_len, return_tensors='pt', truncation=True)
        target = self.tokenizer.batch_encode_plus([ctext], pad_to_max_length=True, max_length=self.target_len, return_tensors='pt', truncation=True)

        if index < 5:
            logger.info("Source: {}".format(self.tokenizer.batch_decode(source['input_ids'])))
            logger.info("Target: {}".format(self.tokenizer.batch_decode(target['input_ids'])))

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'input_ids': source_ids.to(dtype=torch.long),
            'attention_mask': source_mask.to(dtype=torch.long),
            'encoder_hidden_states': target_ids.to(dtype=torch.long),
            'encoder_attention_mask': target_mask.to(dtype=torch.long)
        }