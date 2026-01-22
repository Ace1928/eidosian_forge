import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
def text_infilling(self, prefixes: List[str], suffixes: List[str], temperature: float=0.6, top_p: float=0.9, max_gen_len: Optional[int]=None, logprobs: bool=False, suffix_first: bool=False) -> List[InfillingPrediction]:
    assert self.tokenizer.eot_id is not None
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = [infilling_prompt_tokens(self.tokenizer, prefix, suffix, suffix_first=suffix_first) for prefix, suffix in zip(prefixes, suffixes)]
    generation_tokens, generation_logprobs = self.generate(prompt_tokens=prompt_tokens, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs, echo=False, stop_token=self.tokenizer.eot_id)
    generations = [self.tokenizer.decode_infilling(t) for t in generation_tokens]
    if logprobs:
        assert generation_logprobs is not None
        return [{'generation': generation, 'logprobs': logprobs_i, 'tokens': [self.tokenizer.token_piece(x) for x in t], 'full_text': prefix + generation + suffix} for prefix, suffix, generation, t, logprobs_i in zip(prefixes, suffixes, generations, generation_tokens, generation_logprobs)]
    else:
        return [{'generation': generation, 'full_text': prefix + generation + suffix} for prefix, suffix, generation in zip(prefixes, suffixes, generations)]