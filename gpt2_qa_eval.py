import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AutoConfig
from transformers import OPTForCausalLM
from torch.utils.data.dataloader import default_collate
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
from collections import defaultdict
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


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

from collections import Counter

def calculate_rouge_n_score(references, candidates, n=1):
    scores = []
    for ref, cand in zip(references, candidates):
        ref_ngrams = Counter(ngrams(word_tokenize(ref), n))
        cand_ngrams = Counter(ngrams(word_tokenize(cand), n))

        # Calculate recall
        overlap = sum((ref_ngrams & cand_ngrams).values())
        total_ref_ngrams = sum(ref_ngrams.values())
        recall = overlap / total_ref_ngrams if total_ref_ngrams > 0 else 0
        scores.append(recall)

    # Calculate average score
    return sum(scores) / len(scores) if scores else 0

# Helper function to generate n-grams
def ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
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

evalbatch_size=64

print("Loading LM Extraction eval dataset")
evaldataset,train_preprefix,train_dataset=load_lmdataset()
preprocessed_data = preprocess_dataset(evaldataset)
# evaldata_loader = DataLoader(evaldataset, batch_size=evalbatch_size, shuffle=False)
evaldata_loader = DataLoader(preprocessed_data, batch_size=evalbatch_size, shuffle=False)

epochs = 6  # Number of epochs
model_base_name = "./gpt2-qa/checkpoint-"
# tokenizer_base_name='./models/gpt2tok_cnn_dailymail_epoch'
model_names=[1,2,3,4,5,6]
steps=122

bleu_scores={}
rg_scores={}
for epoch in range(0, epochs):
    model_name = f"{model_base_name}{str(steps*model_names[epoch])}"
    # tokenizer_name = f"{tokenizer_base_name}{epoch}"
    
    print("Loading model",model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    bleu_scores[epoch]=[]
    rg_scores[epoch]=[]
    
    total_bleu_score=0
    total_rg_score=0
    total_samples=0

    model.eval()
    with torch.no_grad():
      for i, batch in enumerate(tqdm(evaldata_loader,desc="Memorization Loop")):

          input_len = 10
          # prompts = []
          # input_ids = []
          # attention_mask = []
          prefixes, true_suffixes = batch
          # print("prefixes",prefixes)
          # print("suffixes ",true_suffixes)
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
        #   print("gen text",generated_suffixes)


          bleu_score = calculate_bleu_score(true_suffixes, generated_suffixes)
          bleu_scores[epoch].append(bleu_score)
        #   print(bleu_score)
          total_bleu_score += bleu_score
          
          rg_score = calculate_rouge_n_score(true_suffixes, generated_suffixes)
          rg_scores[epoch].append(rg_score)
        #   print(rg_score)
    
          total_rg_score += rg_score

          # print(generated_suffixes)
          # bert_score = calculate_bertscore(true_suffixes, generated_suffixes)
          # total_bert_score += bert_score
          # bert_scores[epoch].append(bert_score)
          # # print(f'Batch {i} bleu score {bleu_score}')
          total_samples+=1
    
    avg_bleu_score = total_bleu_score / total_samples
    # bleu_scores.append(avg_bleu_score)
    print(f'avg bleu scores : {avg_bleu_score},total_score:{total_bleu_score}')
    
    avg_rg_score = total_rg_score / total_samples
    # bleu_scores.append(avg_bleu_score)
    print(f'avg rouge-1 scores : {avg_rg_score},total_score:{total_rg_score}')

    # avg_bert_score = total_bert_score / total_samples
    # bert_scores.append(avg_bert_score)
    # print(f'bert scores : {bert_scores}')


# np.save("bert_scores_dailymail.npy", np.array(bleu_scores))

    
epochs = list(bleu_scores.keys())
scores = np.array(list(bleu_scores.values()))
structured_array = np.column_stack((epochs, scores))
np.save("bleu_scores_qa_gpt2_final.npy", structured_array)

epochs = list(rg_scores.keys())
scores = np.array(list(rg_scores.values()))
structured_array = np.column_stack((epochs, scores))
np.save("rouge1_scores_qa_gpt2_final.npy", structured_array)