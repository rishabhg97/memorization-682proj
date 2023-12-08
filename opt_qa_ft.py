from torch.utils.data.dataloader import default_collate
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
from collections import defaultdict
from transformers import GPT2Tokenizer, AutoTokenizer
from torch.utils.data import DataLoader
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AutoConfig
from transformers import OPTForCausalLM
from datasets import Dataset
CACHE_DIR = "./cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
# Load dataset
raw_dataset = load_dataset("squad", cache_dir=CACHE_DIR)
# raw_dataset = load_dataset("cnn_dailymail","3.0.0", cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m",use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'left'


# prefix='summarise:'

def preprocess_function(examples):
    # print(examples)
    questions=examples['question']
    texts=examples['context']
    answers = [answer['text'][0] if answer['text'] else '' for answer in examples['answers']]

    answers = examples['answers'][0]['text'][0] if examples['answers'][0]['text'] else ''
    input_texts = [f"context: {context} question: {question} answer: {answer}" 
                   for context, question, answer in zip(texts, questions, answers)]

    # Tokenizing the input text
    return tokenizer(input_texts, max_length=512, padding="max_length", truncation=True, return_attention_mask=True)

# special_tokens = tokenizer.special_tokens_map

# total_size = len(raw_dataset['train'])
# subset_size = int(0.2 * total_size)
# random_indices = np.random.permutation(total_size)[:subset_size]
# train_subset = raw_dataset['train'].select(random_indices)
tokenized_train_dataset=raw_dataset['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset["train"].column_names
)
# total_size = len(raw_dataset['validation'])
# subset_size = int(0.2 * total_size)
# random_indices = np.random.permutation(total_size)[:subset_size]
# val_subset = raw_dataset['validation'].select(random_indices)
tokenized_val_dataset=raw_dataset['validation'].map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset["validation"].column_names
)
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

model =OPTForCausalLM.from_pretrained("facebook/opt-350m")
# model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

training_args = TrainingArguments(
    output_dir="./opt_qa_models",          # output directory
    num_train_epochs=6,              # total # of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    # warmup_steps=200,                # number of warmup steps for learning rate scheduler
    # weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    prediction_loss_only=True,
    evaluation_strategy="epoch", # Save model at the end of each epoch
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_strategy="epoch",
)
from transformers import TrainerCallback
class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        output_dir = f"./opt-finetuned-cnn-summ/epoch_{state.epoch}"
        model.save_pretrained(output_dir)



trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset
)

trainer.train()