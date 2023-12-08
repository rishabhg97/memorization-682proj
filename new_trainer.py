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
from datasets import Dataset
CACHE_DIR = "./cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)



# Load dataset
raw_dataset = load_dataset("cnn_dailymail","3.0.0", cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained("gpt2",use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'


prefix='summarise:'

def preprocess_function(examples):
  inputs = [ prefix + doc for doc in examples['article'] ]
  model_inputs = tokenizer(inputs, max_length=512,truncation=True,padding=True)

  labels = tokenizer(text_target=examples['highlights'], max_length=128,truncation=True,padding='max_length')
  
  model_inputs['labels'] = labels['input_ids']
  
  if not all(len(label) == 128 for label in model_inputs['labels']):
    print("Inconsistent label lengths found")
  
  return model_inputs
# special_tokens = tokenizer.special_tokens_map
train_batch_size=32
val_batch_size=16
tokenized_train_dataset=raw_dataset['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset["train"].column_names
)
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
# model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained("gpt2",use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

training_args = TrainingArguments(
    output_dir="./models",          # output directory
    num_train_epochs=6,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    # warmup_steps=200,                # number of warmup steps for learning rate scheduler
    # weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    prediction_loss_only=True,
    evaluation_strategy="epoch", # Save model at the end of each epoch
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_strategy="epoch",
    save_strategy="no"
)
from transformers import TrainerCallback
class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        output_dir = f"./gpt2-finetuned-cnn-summ/epoch_{state.epoch}"
        model.save_pretrained(output_dir)



trainer = Trainer(
    model=model,                         
    args=training_args,                 
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,         
    eval_dataset=tokenized_val_dataset            
)

trainer.train()
def test_model(model, tokenizer, test_texts):
    model.eval()  # Put the model in evaluation mode
    for text in test_texts:
        # Encode the text for input
        inpt="summarize:" + text
        inputs = tokenizer.encode(inpt, return_tensors='pt', truncation=True, max_length=512).to(device)

        # Generate the summary
        summary_ids = model.generate(inputs, max_length=100, min_length=len(inputs), length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary=summary[len(inpt):]
        print(f"Original Text: {text}\nSummary: {summary}\n")

# Example usage
test_texts = [
    "The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled 'Computing Machinery and Intelligence' which proposed what is now called the Turing Test as a criterion of intelligence.",
]
test_model(model, tokenizer, test_texts)
