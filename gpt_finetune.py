import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer,DataCollatorForLanguageModeling
from torch.utils.data.dataloader import default_collate

from datasets import load_dataset
from tqdm import tqdm
import os
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
train_batch_size=16
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

# sample_data = dataset['train'].select(range(50))


# Tokenize dataset
tokenized_dataset_path = os.path.join(CACHE_DIR, "tokenized_dataset.pt")
if os.path.exists(tokenized_dataset_path):
    tokenized_datasets = torch.load(tokenized_dataset_path)
else:
    # Tokenize and cache dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True,load_from_cache_file=True)
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

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
# DataLoader
data_collator = SummarizationDataCollator()
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, 
#     mlm=False, 
# )
# for idx, data in enumerate(tokenized_datasets['train']):
#     print(data.keys())
    
    
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=train_batch_size,collate_fn=data_collator)
val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=val_batch_size,collate_fn=data_collator)

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3

# for i, batch in enumerate(train_dataloader):
#     print(f"Batch {i}: input_ids shape - {batch['input_ids'].shape}, attention_mask shape - {batch['attention_mask'].shape}")
#     if i >= 2:  # Inspect only the first few batches
#         break


for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = batch["input_ids"]
        labels = batch['labels']
        outputs = model( inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save checkpoint
    torch.save(model.state_dict(), f"gpt2_cnn_dailymail_epoch{epoch}.pt")

    # Evaluation (optional)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()
            outputs = model(inputs, labels=labels)
            total_loss += outputs.loss.item()

    print(f"Validation Loss after Epoch {epoch}: {total_loss / len(val_dataloader)}")

# Save final model
torch.save(model.state_dict(), "gpt2_cnn_dailymail_final.pt")
