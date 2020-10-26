# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from mydataset import MyDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

# Print for allenai beaker verification
print(device)
print(torch.cuda.device_count())

TRAIN_DATA_PATH = '/home/danielk/nqopen_csv/train.tsv'
DEV_DATA_PATH = '/home/danielk/nqopen_csv/dev.tsv'
model_name = 't5-small'
IN_LEN = 20
OUT_LEN = 10


def main():
    wandb.init(project="finetuning_t5")

    config = wandb.config
    torch.manual_seed(103)  # pytorch random seed
    np.random.seed(103)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    train_dataset = pd.read_csv(TRAIN_DATA_PATH, encoding='latin-1', sep="\t")
    train_dataset = train_dataset[['question', 'ans', 'all_ans']]
    print(train_dataset.head())

    val_dataset = pd.read_csv(DEV_DATA_PATH, encoding='latin-1', sep="\t")
    val_dataset = val_dataset[['question', 'ans', 'all_ans']]
    print(val_dataset.head())

    print("TRAIN Dataset tuple count: {}".format(train_dataset.shape))
    print("DEV Dataset tuple_count: {}".format(val_dataset.shape))

    training_set = MyDataset(train_dataset, tokenizer, IN_LEN, OUT_LEN)
    val_set = MyDataset(val_dataset, tokenizer, IN_LEN, OUT_LEN)

    model = T5ForConditionalGeneration.from_pretrained(model_name, use_cdn=False)
    model = model.to(device)

    # optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    wandb.watch(model, log="all")
    print('Initiating Fine-Tuning for the model on our dataset')

    training_args = {
        'max_len': config.IN_LEN,
        'target_max_len': config.SUMMARY_LEN,
        'device': device,
        "max_len": 512,
        "target_max_len": 16,
        "output_dir": './models/',
        "overwrite_output_dir": True,
        "per_gpu_train_batch_size": 32,
        "per_gpu_eval_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        # "tpu_num_cores": 8,
        "num_train_epochs": 20,
        "do_train": True
    }
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=val_set,
        prediction_loss_only=True,
    )

    for epoch in range(config.TRAIN_EPOCHS):
        trainer.train()
        trainer.save_model()
        # if trainer.is_world_master():
        #     tokenizer.savePretrained('/models')
        # train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader_mini)

    model.save_pretrained('models')


if __name__ == '__main__':
    main()
