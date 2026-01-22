import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
def text_completion(self, prompts: List[str], temperature: float=0.6, top_p: float=0.9, max_gen_len: Optional[int]=None, logprobs: bool=False, echo: bool=False) -> List[CompletionPrediction]:
    """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    generation_tokens, generation_logprobs = self.generate(prompt_tokens=prompt_tokens, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs, echo=echo)
    if logprobs:
        return [{'generation': self.tokenizer.decode(t), 'tokens': [self.tokenizer.decode([x]) for x in t], 'logprobs': logprobs_i} for t, logprobs_i in zip(generation_tokens, generation_logprobs)]
    return [{'generation': self.tokenizer.decode(t)} for t in generation_tokens]