from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os

CACHE_DIR = "./.cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def load_dataset_and_tokenize(tokenizer, dataset_name="cnn_dailymail", block_size=512):
    def tokenize_function(examples):
        return tokenizer(examples["article"], truncation=True, max_length=block_size)

    dataset = load_dataset(dataset_name, '3.0.0')
    tokenized_datasets = dataset.map(tokenize_function, batched=True,load_from_cache_file=True)
    return tokenized_datasets

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenized_datasets = load_dataset_and_tokenize(tokenizer)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-cnn_dailymail",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

if __name__ == "__main__":
    main()
