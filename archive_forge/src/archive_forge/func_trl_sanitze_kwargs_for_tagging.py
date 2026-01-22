import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from accelerate import PartialState
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from ..import_utils import is_peft_available, is_unsloth_available, is_xpu_available
from ..trainer.model_config import ModelConfig
def trl_sanitze_kwargs_for_tagging(model, tag_names, kwargs=None):
    if is_unsloth_available():
        if hasattr(model, 'config') and getattr(model.config, 'unsloth_version', None) is not None:
            tag_names.append('unsloth')
    if kwargs is not None:
        if 'tags' not in kwargs:
            kwargs['tags'] = tag_names
        elif 'tags' in kwargs and isinstance(kwargs['tags'], list):
            kwargs['tags'].extend(tag_names)
        elif 'tags' in kwargs and isinstance(kwargs['tags'], str):
            tag_names.append(kwargs['tags'])
            kwargs['tags'] = tag_names
    return kwargs