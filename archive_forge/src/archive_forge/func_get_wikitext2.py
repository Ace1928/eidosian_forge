import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets import load_dataset
def get_wikitext2(tokenizer: Any, seqlen: int, nsamples: int, split: str='train'):
    if split == 'train':
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    elif split == 'validation':
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = ''.join([' \n' if s == '' else s for s in data['text'][:1000]])
    enc = tokenizer(text, return_tensors='pt')
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({'input_ids': inp, 'attention_mask': attention_mask})
    return dataset