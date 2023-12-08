import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2ForQuestionAnswering
from transformers import (
    GPT2LMHeadModel,
    GPT2ForQuestionAnswering,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from torch.utils.data.dataloader import default_collate
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
from collections import defaultdict
import logging
import datetime

# Configure logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f'./logs/logfile_{timestamp}.log', level=logging.INFO, format='%(asctime)s: %(message)s')


import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class QADataCollator:
    def __call__(self, batch):
        # Convert each item in the batch to tensors and stack them
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
        labels = torch.stack([torch.tensor(item['labels']) for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


CACHE_DIR = "./cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)



# Load dataset
raw_dataset = load_dataset("squad", cache_dir=CACHE_DIR)
# Define constants /args
title_column = 'title'
text_column ='context'
question_column = 'question'
answer_column = 'answers'
max_source_length=100
max_target_length=100
ignore_pad_token_for_loss=True
train_batch_size=48
val_batch_size=16

class SquadDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        context = example['context']
        question = example['question']
        if len(example['answers']['text'])>0:
          answer = example['answers']['text'][0]
        else:
          answer=''
        
        # do encoding of the context and question 
        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='max_length',   
            max_length=384,
            truncation=True
        )
        
        # get start and end positions of answer in input_ids
        input_ids = encoding['input_ids']
        answer_start = example['answers']['answer_start'][0]
        answer_end = answer_start + len(answer)
        
        start_positions = []
        end_positions = []
        for i, token_id in enumerate(input_ids):
            if i == answer_start:
                start_positions.append(i)
            else:
                start_positions.append(-100)
            
            if i == answer_end:
                end_positions.append(i)
            else:
                end_positions.append(-100)
        
        # Create input tensors
        inputs = {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoding['token_type_ids'], dtype=torch.long),
            'start_positions': torch.tensor(start_positions, dtype=torch.float),  # start and end positions should be float
            'end_positions': torch.tensor(end_positions, dtype=torch.float)
        }
        
        return inputs, answer

from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("gpt2",use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
# special_tokens = tokenizer.special_tokens_map

# Create the Dataloader
train_dataloader = DataLoader(
    SquadDataset(raw_dataset['train'], tokenizer),
    batch_size=train_batch_size,
    shuffle=True
)
val_dataloader = DataLoader(
    SquadDataset(raw_dataset['validation'], tokenizer),
    batch_size=val_batch_size,
    shuffle=True
)


# Tokenize dataset
# tokenized_dataset_path = os.path.join(CACHE_DIR, "gpt2tokenized_dataset_squad.pt")
# print(f'Tokenized dataset path {tokenized_dataset_path}')
# # print(tokenized_dataset_path)
# def add_end_token_to_question(input_dict):
#     input_dict['question'] += special_tokens['bos_token']
#     return input_dict
# dataset = dataset.map(add_end_token_to_question)

# if os.path.exists(tokenized_dataset_path):
#     tokenized_datasets = torch.load(tokenized_dataset_path)
# # else:
#     # Tokenize and cache dataset

# print("Dataset Columns and Keys:")
# print(tokenized_datasets)

class MemorisationDataset(Dataset):
    def __init__(self, prefix_file, suffix_file):
        self.prefixes = np.load(prefix_file).astype(np.int64)
        self.suffixes = np.load(suffix_file).astype(np.int64)

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        return self.prefixes[idx], self.suffixes[idx]


def load_lmdataset():
    logging.info(f'Loading dataset from ./data/ folder')

    # print()
    # train_prefix = np.load('./data/train_prefix.npy')
    # train_suffix = np.load('./data/train_suffix.npy')
    train_preprefix = np.load('./data/train_preprefix.npy')
    train_dataset = np.load('./data/train_dataset.npy')
    dataset = MemorisationDataset('./data/train_prefix.npy', './data/train_suffix.npy')

    return dataset,train_preprefix,train_dataset


def calculate_bleu_score(references, candidates):
    score = 0
    for ref, cand in zip(references, candidates):
        ref_tokens = [word_tokenize(ref)]
        cand_tokens = word_tokenize(cand)
        score += sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    return score / len(references)

def load_tokenizer_for_causal_lm(model_name):
    """
    Load tokenizer with required config changes
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For Autoregressive models, padding on the right would mean the model
    # will receive padded tokens as context, which is not useful during generation
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

logging.info("Using device: %s", device)

# DataLoader
data_collator = QADataCollator()
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,
# )
# for idx, data in enumerate(tokenized_datasets['train']):
#     print(data.keys())


# train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=16,collate_fn=data_collator)
# val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=val_batch_size,collate_fn=data_collator)

# Load model
model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 5


# for i, batch in enumerate(train_dataloader):
#     print(f"Batch {i}: input_ids shape - {batch['input_ids'].shape}, attention_mask shape - {batch['attention_mask'].shape}")
#     if i >= 2:  # Inspect only the first few batches
#         break

tokenizer_gpt2 = load_tokenizer_for_causal_lm("gpt2")
print("Loaded tokenizer for mem dataset",tokenizer_gpt2)

def preprocess_dataset(dataset):
    decoded_prefixes = [tokenizer_gpt2.decode(prefix) for prefix in dataset.prefixes]
    decoded_suffixes = [tokenizer_gpt2.decode(suffix) for suffix in dataset.suffixes]
    return list(zip(decoded_prefixes, decoded_suffixes))


# Top k sampling
top_k = 40
max_length_prefix=50
max_length_suffix=50
evalbatch_size=16
logging.info("Loading LM Extraction eval dataset")
evaldataset,train_preprefix,train_dataset=load_lmdataset()
preprocessed_data = preprocess_dataset(evaldataset)
# evaldata_loader = DataLoader(evaldataset, batch_size=evalbatch_size, shuffle=False)
evaldata_loader = DataLoader(preprocessed_data, batch_size=evalbatch_size, shuffle=False)

def train_loop(dataloader, model, optimizer):
    
    # set the model to training model
    model.train()
    
    for i,batch in enumerate(tqdm(dataloader,desc="Train Loop")):
        optimizer.zero_grad()
        
        # previous tokens
        input_ids = batch[0]['input_ids'].to(device)
        attention_mask = batch[0]['attention_mask'].to(device)
        token_type_ids = batch[0]['token_type_ids'].to(device)
        start_positions = batch[0]['start_positions'].to(device)
        end_positions = batch[0]['end_positions'].to(device)
        
        labels = {
            'start_positions': start_positions,
            'end_positions': end_positions
        }
        
       # get outputs from model
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # calculate loss
        loss_start = nn.CrossEntropyLoss()(outputs.start_logits, start_positions)
        loss_end = nn.CrossEntropyLoss()(outputs.end_logits, end_positions)
        loss = (loss_start + loss_end) / 2  # average loss for start and end positions
        
        # backpropagation
        loss.backward()
        optimizer.step()
        

def val_loop(dataloader, model):
    # set the model of evaluation
    model.eval()
    val_loss = 0
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for i,batch in enumerate(tqdm(dataloader,desc="Val Loop")):
            # previous tokens
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            token_type_ids = batch[0]['token_type_ids'].to(device)
            start_positions = batch[0]['start_positions'].to(device)
            end_positions = batch[0]['end_positions'].to(device)

            labels = {
                'start_positions': start_positions,
                'end_positions': end_positions
            }

           # get outputs from model
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            # calculate loss
            loss_start = nn.CrossEntropyLoss()(outputs.start_logits, start_positions)
            loss_end = nn.CrossEntropyLoss()(outputs.end_logits, end_positions)
            loss = (loss_start + loss_end) / 2  # average loss for start and end positions
            
            val_loss += loss.item()
    
    # Print the validation loss for this epoch
    
    logging.info(f"Validation Loss: {val_loss/len(dataloader)}")
    
import transformers
import torch.nn as nn
transformers.logging.set_verbosity_error()
bleu_scores = {}
test_iters=10
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}\n ---------------------------")
    train_loop(train_dataloader, model, optimizer)
     # Save checkpoint
    model.save_pretrained(f"./models/gpt2_squad_epoch{epoch}")
    tokenizer.save_pretrained(f"./models/gpt2tok_squad_epoch{epoch}")
    model.eval()
    lm_model.load_state_dict(model.state_dict(), strict=False)
    model.to('cpu')
    lm_model.to(device)
    total_bleu_score=0
    total_samples=0
    bleu_scores[epoch]=[]
    with torch.no_grad():
      for i, batch in enumerate(tqdm(evaldata_loader,desc="Memorization Loop")):
          input_len = 10
          # prompts = []
          # input_ids = []
          # attention_mask = []
          prefixes, true_suffixes = batch
          # decoded_prefixes = [tokenizer_gpt2.decode(prefix) for prefix in prefixes.numpy()]
          # decoded_true_suffixes = [tokenizer_gpt2.decode(suffix) for suffix in true_suffixes.numpy()]

          inputs = tokenizer(prefixes, return_tensors='pt', padding=True).to(device)

          generated_sequences = lm_model.generate(
              input_ids = inputs['input_ids'],
              attention_mask = inputs['attention_mask'],
              pad_token_id=tokenizer.eos_token_id,
              max_length = max_length_prefix+max_length_suffix,
              do_sample = True,
              top_k = top_k,
              top_p = 1.0
          )
          generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

          generated_suffixes = [text[len(prefix):] for text, prefix in zip(generated_texts,prefixes)]

          bleu_score = calculate_bleu_score(true_suffixes, generated_suffixes)
          total_bleu_score += bleu_score
          bleu_scores[epoch].append(bleu_score)
          # print(f'Batch {i} bleu score {bleu_score}')
          total_samples+=1
    lm_model.to('cpu')
    model.to(device)

    avg_bleu_score = total_bleu_score / total_samples
    # bleu_scores.append(avg_bleu_score)
    logging.info(f'bleu scores : {total_bleu_score}, avg bleu score :{avg_bleu_score}')
    val_loop(val_dataloader, model)

epochs = list(bleu_scores.keys())
scores = np.array(list(bleu_scores.values()))
structured_array = np.column_stack((epochs, scores))
np.save("bleu_scores_squad.npy", structured_array)

logging.info("Done!")