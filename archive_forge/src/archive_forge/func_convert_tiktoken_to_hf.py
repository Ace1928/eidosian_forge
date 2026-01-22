import argparse
import io
import json
import os
import tempfile
import urllib
import warnings
from typing import Any, Optional, Tuple
import torch
from huggingface_hub.utils import insecure_hashlib
from torch import nn
from tqdm import tqdm
from transformers import (
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
from transformers.utils.import_utils import _is_package_available
def convert_tiktoken_to_hf(multilingual: bool=True, num_languages: int=100, time_precision=0.02) -> WhisperTokenizer:
    tiktoken_tokenizer_path = _TOKENIZERS['multilingual' if multilingual else 'english']
    start_of_transcript = ['<|endoftext|>', '<|startoftranscript|>']
    control_tokens = ['<|translate|>', '<|transcribe|>', '<|startoflm|>', '<|startofprev|>', '<|nospeech|>', '<|notimestamps|>']
    language_tokens = [f'<|{k}|>' for k in list(LANGUAGES)[:num_languages]]
    timestamp_tokens = ['<|%.2f|>' % (i * time_precision) for i in range(1500 + 1)]
    vocab, merges = convert_tiktoken_bpe_to_hf(tiktoken_tokenizer_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_file = f'{tmpdirname}/vocab.json'
        merge_file = f'{tmpdirname}/merges.txt'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(vocab, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        with open(merge_file, 'w', encoding='utf-8') as writer:
            writer.write('#version: 0.2\n')
            for bpe_tokens in merges:
                writer.write(bpe_tokens + '\n')
        hf_tokenizer = WhisperTokenizer(vocab_file, merge_file)
    hf_tokenizer.add_tokens(start_of_transcript + language_tokens + control_tokens, special_tokens=True)
    hf_tokenizer.add_tokens(timestamp_tokens, special_tokens=False)
    return hf_tokenizer