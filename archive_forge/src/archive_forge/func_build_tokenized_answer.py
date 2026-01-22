import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import (
def build_tokenized_answer(self, prompt, answer):
    """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
    full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
    answer_input_ids = full_tokenized['input_ids'][len(prompt_input_ids):]
    answer_attention_mask = full_tokenized['attention_mask'][len(prompt_input_ids):]
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
    full_input_ids = np.array(full_tokenized['input_ids'])
    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError('Prompt input ids and answer input ids should have the same length.')
    response_token_ids_start_idx = len(prompt_input_ids)
    if prompt_input_ids != full_tokenized['input_ids'][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1
    prompt_input_ids = full_tokenized['input_ids'][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized['attention_mask'][:response_token_ids_start_idx]
    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError('Prompt input ids and attention mask should have the same length.')
    answer_input_ids = full_tokenized['input_ids'][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized['attention_mask'][response_token_ids_start_idx:]
    return dict(prompt_input_ids=prompt_input_ids, prompt_attention_mask=prompt_attention_mask, input_ids=answer_input_ids, attention_mask=answer_attention_mask)