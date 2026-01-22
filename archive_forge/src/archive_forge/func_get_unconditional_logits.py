import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def get_unconditional_logits(self, input_ids):
    if self.unconditional_context['first_pass']:
        if self.unconditional_context['input_ids'] is None:
            self.unconditional_context['input_ids'] = input_ids[:, -1:]
        if self.unconditional_context['attention_mask'] is None:
            self.unconditional_context['attention_mask'] = torch.ones_like(self.unconditional_context['input_ids'], dtype=torch.long)
        input_ids = self.unconditional_context['input_ids']
        attention_mask = self.unconditional_context['attention_mask']
        self.unconditional_context['first_pass'] = False
    else:
        attention_mask = torch.cat([self.unconditional_context['attention_mask'], torch.ones_like(input_ids[:, -1:], dtype=torch.long)], dim=1)
        if not self.unconditional_context['use_cache']:
            input_ids = torch.cat([self.unconditional_context['input_ids'], input_ids[:, -1:]], dim=1)
        else:
            input_ids = input_ids[:, -1:]
        self.unconditional_context['input_ids'] = input_ids
        self.unconditional_context['attention_mask'] = attention_mask
    out = self.model(input_ids, attention_mask=attention_mask, use_cache=self.unconditional_context['use_cache'], past_key_values=self.unconditional_context['past_key_values'])
    self.unconditional_context['past_key_values'] = out.get('past_key_values', None)
    return out.logits