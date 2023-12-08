import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
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


import nltk
# nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize



CACHE_DIR = "./cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

from torch.utils.data.dataloader import default_collate

class SummarizationDataCollator:
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

# Load dataset
dataset = load_dataset("cnn_dailymail", '3.0.0',cache_dir=CACHE_DIR)
# Define constants /args
text_column='article'
summary_column='highlights'
max_source_length=100
max_target_length=100
ignore_pad_token_for_loss=True
train_batch_size=128
val_batch_size=8

# Tokenize function
def preprocess_function(examples):
    # print("Original:", examples)

    inputs = examples[text_column]
    targets = examples[summary_column]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding='max_length', truncation=True)

    # Tokenize targets
    labels = tokenizer(targets, max_length=max_target_length, padding='max_length', truncation=True)
    if ignore_pad_token_for_loss:
        # Replace pad token id (-100) where appropriate
        labels["input_ids"] = [
            label if label != tokenizer.pad_token_id else -100 for label in labels["input_ids"]
        ]
    # Replace padding token id in labels with -100 if ignoring pad token for loss
    # if ignore_pad_token_for_loss:
    #     labels["input_ids"] = [
    #         [(label if label != tokenizer.pad_token_id else -100) for label in label_ids] for label_ids in labels["input_ids"]
    #     ]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"]
    }


tokenizer = GPT2Tokenizer.from_pretrained("gpt2",use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# sample_data = dataset['train'].select(range(50))


# Tokenize dataset
tokenized_dataset_path = os.path.join(CACHE_DIR, "tokenized_dataset.pt")
# print(tokenized_dataset_path)
if os.path.exists(tokenized_dataset_path):
    tokenized_datasets = torch.load(tokenized_dataset_path)
else:
    # Tokenize and cache dataset
  tokenized_datasets = dataset.map(preprocess_function, batched=True,load_from_cache_file=False)
  print(f'Saving tokenized dataset in this path {tokenized_dataset_path}')
  torch.save(tokenized_datasets, tokenized_dataset_path)
# tokenized_datasets = dataset.map()
# for i, example in enumerate(tokenized_datasets):
#     print(f"Example {i}: {example}")
#     if i >= 2:  # Inspect only the first few examples
#         break
print("Dataset Columns and Keys:")
print(tokenized_datasets)
# Print columns for each split (e.g., train, validation, test)
# for split in tokenized_datasets.keys():
#     print(f"\n{split} Split:")
#     # Print column names
#     print("Columns:", tokenized_datasets[split].column_names)

#     # Optionally, print a few example keys (IDs) from the dataset
#     print("Example Keys:", [tokenized_datasets[split][i]['id'] for i in range(3)])

class MemorisationDataset(Dataset):
    def __init__(self, prefix_file, suffix_file):
        self.prefixes = np.load(prefix_file).astype(np.int64)  # Convert to int64
        self.suffixes = np.load(suffix_file).astype(np.int64)

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        return self.prefixes[idx], self.suffixes[idx]


def load_lmdataset():
    print(f'Loading dataset from ./data/ folder')
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


print("Using device:", device)

data_collator = SummarizationDataCollator()

torch.cuda.empty_cache()

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=16,collate_fn=data_collator)
val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=val_batch_size,collate_fn=data_collator)

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
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

print("Loading LM Extraction eval dataset")
evaldataset,train_preprefix,train_dataset=load_lmdataset()
preprocessed_data = preprocess_dataset(evaldataset)
# evaldata_loader = DataLoader(evaldataset, batch_size=evalbatch_size, shuffle=False)
evaldata_loader = DataLoader(preprocessed_data, batch_size=evalbatch_size, shuffle=False)

bleu_scores = []
# test_iters=10

# DataLoader


for epoch in range(epochs):
    model.train()
    for i,batch in enumerate(tqdm(train_dataloader,desc="Train Loop")):
        # if test_iters is not None and i> test_iters:
        #     break
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = batch["input_ids"]
        labels = batch['labels']
        outputs = model( inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    model.save_pretrained(f"./models/gpt2_cnn_dailymail2_epoch{epoch}.pt")
    tokenizer.save_pretrained(f"./models/gpt2tok_cnn_dailymail2_epoch{epoch}.pt")
    # torch.save(model.state_dict(), )

    #Memorization eval
    total_bleu_score=0
    total_samples=0
    model.eval()


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

          generated_sequences = model.generate(
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
          # print(f'Batch {i} bleu score {bleu_score}')
          total_samples+=1

    avg_bleu_score = total_bleu_score / total_samples
    bleu_scores.append(avg_bleu_score)
    print(f'bleu scores : {bleu_scores}')

    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader,desc="Validation Loop"):
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()
            outputs = model(inputs, labels=labels)
            total_loss += outputs.loss.item()

    print(f"Validation Loss after Epoch {epoch}: {total_loss / len(val_dataloader)}")

# Save final model
torch.save(model.state_dict(), "gpt2_cnn_dailymail_final.pt")
np.save("bleu_scores_dailymail.npy", np.array(bleu_scores))
# run and rename the bleu scores file