import argparse
import numpy as np
import sys
import math
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

LOW_MEMORY = True


class MemorisationDataset(Dataset):
    def __init__(self, prefix_file, suffix_file):
        self.prefixes = np.load(prefix_file).astype(np.int64)  # Convert to int64
        self.suffixes = np.load(suffix_file).astype(np.int64)
    
    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        return self.prefixes[idx], self.suffixes[idx]

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

def load_model_for_causal_lm(model_name, device):
    """
    Load model with required config changes
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=LOW_MEMORY).to(device)

    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    return model


def calculate_perplexity(input_sentence, model, tokenizer, device):
    """
    Calculate exp(loss), where loss is obtained py passing tokenized input sentence to the model
    with the labels set as the same tokenized input (the shifting of the labels is done internally)
    https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel.forward.labels
    """
    tokenized = tokenizer(input_sentence)
    input = torch.tensor(tokenized.input_ids).to(device)
    with torch.no_grad():
        output = model(input, labels=input)
    
    return torch.exp(output.loss)

def calculate_perplexity_sliding(input_sentence, model, tokenizer, device, window_size=50):
    """
    Calculate min(exp(loss)) over a sliding window
    """
    tokenized = tokenizer(input_sentence)
    input = torch.tensor(tokenized.input_ids).to(device)
    min_perplexity = 100000
    with torch.no_grad():
        for start_idx in range(input.shape[0]-window_size):
            input_window = input[start_idx: start_idx+window_size]
            output = model(input_window, labels=input_window)
            min_perplexity = min(min_perplexity, torch.exp(output.loss))
    return min_perplexity

def print_best(metric, samples, metric_name, name1, scores1, name2=None, scores2=None, lower_better=True, n=10):
    """
    Print the top-n best samples according to the given metric
    """
    if lower_better:
        idxs = np.argsort(metric)[:n]
    else:
        idxs = np.argsort(metric)[::-1][:n]

    print("Metric Name:", metric_name)
    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        pprint(samples[idx])
        print()
        print()

def print_best_to_file(outfile, metric, samples, metric_name, name1, scores1, name2=None, scores2=None, lower_better=True, n=100):
    """
    Print the top-n best samples according to the given metric to a file
    """
    original_stdout = sys.stdout # Save a reference to the original standard output

    with open(outfile, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("Metric Name:", metric_name)

        if lower_better:
            idxs = np.argsort(metric)[:n]
        else:
            idxs = np.argsort(metric)[::-1][:n]

        for i, idx in enumerate(idxs):
            if scores2 is not None:
                print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
            else:
                print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

            print()
            print(samples[idx])
            print()
            print()
        
        print()
        print()
        sys.stdout = original_stdout # Reset the standard output to its original value

def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines() 
    
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]

    return all_eng

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
    

def main(args):
    # Load models
    model_name='gpt2'
    print(f'Loading models... {model_name}')
    
    tokenizer_gpt2 = load_tokenizer_for_causal_lm("gpt2")
    # MODEL_GPT2 = load_model_for_causal_lm("gpt2", device)
    model=load_model_for_causal_lm(model_name, device)
    tokenizer=load_tokenizer_for_causal_lm(model_name)
    # MODEL_GPT2_MEDIUM = load_model_for_causal_lm("gpt2-medium", device)
    # MODEL_GPT2_XL = load_model_for_causal_lm("gpt2-xl", device)
    # MODEL_GPT2_XL = load_model_for_causal_lm("gpt2-large", device)
    print(f'{model_name} loaded')
    batch_size=32
    max_length_prefix=50
    max_length_suffix=50
    
    # print("Loading commoncrawl...")
    # cc_data = parse_commoncrawl(args.wet_file)
    print("Loading LM Extraction eval dataset")
    evaldataset,train_preprefix,train_dataset=load_lmdataset()
    data_loader = DataLoader(evaldataset, batch_size=batch_size, shuffle=False)
    # print("Checking Lenghts are same")
    # assert len(train_prefix)==len(train_suffix)
    num_batches_to_test=10
    
    # number of tokens to generate (from paper)
    # seq_len = 256

    # k in top_k sampling (from paper)
    top_k = 40
    
    # num_batches = int(math.ceil(args.N / args.batch_size))
    # new_tot = num_batches * args.batch_size
    num_samples=len(data_loader)


    generated_samples = []
    scores = defaultdict(list)
    total_bleu_score=0
    total_samples=0

    for i, batch in enumerate(tqdm(data_loader)):
        if i >= num_batches_to_test:
            break
        input_len = 10
        prompts = []
        input_ids = []
        attention_mask = []
        prefixes, true_suffixes = batch
        decoded_prefixes = [tokenizer_gpt2.decode(prefix) for prefix in prefixes.numpy()]
        decoded_true_suffixes = [tokenizer_gpt2.decode(suffix) for suffix in true_suffixes.numpy()]
        
        inputs = tokenizer(decoded_prefixes, return_tensors='pt', padding=True).to(device)
        # output_sequences = model.generate(
        #     input_ids=inputs['input_ids'],
        #     attention_mask=inputs['attention_mask'],
        #     max_length=max_length + inputs['input_ids'].shape[-1],
        #     temperature=1.0,
        #     top_p=0.9,
        #     pad_token_id=tokenizer.eos_token_id
        # )
        # while len(input_ids) < args.batch_size:
        #     # take some random words in common crawl
        #     r = np.random.randint(0, len(cc_data))
        #     # prompt = " ".join(cc_data[r:r+100].split(" ")[1:-1])
        #     # prompt
        #     # print(prompt)
        #     # make sure we get the same number of tokens for each prompt to enable batching
        #     inputs = TOKENIZER_GPT2(prompt, return_tensors="pt", max_length=input_len, truncation=True)
        #     if len(inputs['input_ids'][0]) == input_len:
        #         input_ids.append(inputs['input_ids'][0])
        #         attention_mask.append(inputs['attention_mask'][0])

        # inputs = {'input_ids': torch.stack(input_ids).to(device), 
        #             'attention_mask': torch.stack(attention_mask).to(device)}

        # the actual truncated prompts (not needed)
        # prompts = TOKENIZER_GPT2.batch_decode(inputs['input_ids'], skip_special_tokens=True)

        # Batched sequence generation
        generated_sequences = model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            max_length = max_length_prefix+max_length_suffix,
            do_sample = True,
            top_k = top_k,
            top_p = 1.0
        )

        generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

        generated_suffixes = [text[len(prefix):] for text, prefix in zip(generated_texts, decoded_prefixes)]

        bleu_score = calculate_bleu_score(decoded_true_suffixes, generated_suffixes)
        total_bleu_score += bleu_score
        print(f'Batch {i} bleu score {bleu_score}')
        total_samples+=1
        
    
    
    average_bleu_score = total_bleu_score / total_samples
    print(f"Average BLEU Score: {average_bleu_score:.2f}")



    # print(len(scores["XL"]))
    # scores["XL"] = np.asarray(scores["XL"])
    # scores["SMALL"] = np.asarray(scores["SMALL"])
    # # scores["MEDIUM"] = np.asarray(scores["MEDIUM"])
    # scores["ZLIB"] = np.asarray(scores["ZLIB"])
    # scores["LOWER"] = np.asarray(scores["LOWER"])
    # scores["WINDOW"] = np.asarray(scores["WINDOW"])

    # # Remove duplicate samples
    # idxs = pd.Index(generated_samples)
    # idxs_mask = ~(idxs.duplicated())
    # print(idxs_mask)
    # generated_samples_clean = np.asarray(generated_samples)[idxs_mask]
    # generated_samples_clean = generated_samples_clean.tolist()
    # scores["XL"] = scores["XL"][idxs_mask]
    # scores["SMALL"] = scores["SMALL"][idxs_mask]
    # # scores["MEDIUM"] = scores["MEDIUM"][idxs_mask]
    # scores["ZLIB"] = scores["ZLIB"][idxs_mask]
    # scores["LOWER"] = scores["LOWER"][idxs_mask]
    # scores["WINDOW"] = scores["WINDOW"][idxs_mask]

    # assert len(generated_samples_clean) == len(scores["XL"])
    # assert len(scores["SMALL"]) == len(scores["XL"])
    # print("Num duplicates:", len(generated_samples) - len(generated_samples_clean))
    
    # # Show best samples based on Metrics
    # # Sort by perplexity of GPT2-XL
    # metric = np.log(scores["XL"])
    # print(f"======== top samples by XL perplexity: ========")
    # print_best(metric, generated_samples_clean, "Sort by perplexity of GPT2-XL", "PPL-XL", scores["XL"], lower_better=True)
    # print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by perplexity of GPT2-XL", "PPL-XL", scores["XL"], lower_better=True)
    # print()
    # print()

    # # Sort by ratio of perplexity of GPT2-XL and GPT2-Small
    # metric = np.log(scores["XL"]) / np.log(scores["SMALL"])
    # print(f"======== top samples by ratio of XL and SMALL perplexity: ========")
    # print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Small", "PPL-XL", scores["XL"], "PPL-SMALL", scores["SMALL"], lower_better=True)
    # print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Small", "PPL-XL", scores["XL"], "PPL-SMALL", scores["SMALL"], lower_better=True)
    # print()
    # print()

    # # Sort by ratio of perplexity of GPT2-XL and GPT2-Medium
    # # metric = np.log(scores["XL"]) / np.log(scores["MEDIUM"])
    # # print(f"======== top samples by ratio of XL and SMALL perplexity: ========")
    # # print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Medium", "PPL-XL", scores["XL"], "PPL-MEDIUM", scores["MEDIUM"], lower_better=True)
    # # print_best_to_file(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Medium", "PPL-XL", scores["XL"], "PPL-MEDIUM", scores["MEDIUM"], lower_better=True)
    # # print()
    # # print()

    # # Sort by ratio of XL perplexity and ZLIB entropy
    # metric = np.log(scores["XL"]) / np.log(scores["ZLIB"])
    # print(f"======== top samples by ratio of XL perplexity and ZLIB entropy: ========")
    # print_best(metric, generated_samples_clean, "Sort by ratio of XL perplexity and ZLIB entropy", "PPL-XL", scores["XL"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
    # print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of XL perplexity and ZLIB entropy", "PPL-XL", scores["XL"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
    # print()
    # print()

    # # Sort by ratio of perplexity of GPT2-XL on normal and lower-cased sample
    # metric = np.log(scores["XL"]) / np.log(scores["LOWER"])
    # print(f"======== top samples by ratio of perplexity of GPT2-XL on normal and lower-cased sample: ========")
    # print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL on normal and lower-cased sample", "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["LOWER"], lower_better=True)
    # print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL on normal and lower-cased sample", "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["LOWER"], lower_better=True)
    # print()
    # print()

    # # Sort by minimum perplexity of GPT2-XL on window of size 50
    # metric = np.log(scores["WINDOW"])
    # print(f"======== top samples by minimum XL perplexity across a sliding window of size 50: ========")
    # print_best(metric, generated_samples_clean, "Sort by minimum perplexity of GPT2-XL on window of size 50", "PPL-WINDOW", scores["WINDOW"], lower_better=True)
    # print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by minimum perplexity of GPT2-XL on window of size 50", "PPL-WINDOW", scores["WINDOW"], lower_better=True)
    # print()
    # print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wet-file', type=str, help='Path to Commoncrawl WET file')
    parser.add_argument('--N', default=20, type=int, help='Number of samples to generate')
    parser.add_argument('--batch_size', default=6, type=int, help='Batch size')
    parser.add_argument('--outfile', type=str, help='Output file to log top samples based on each metric')

    args = parser.parse_args()

    main(args)