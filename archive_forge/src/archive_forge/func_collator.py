import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
from transformers.trainer_utils import EvalLoopOutput
from ..core import PPODecorators
from ..import_utils import is_peft_available
def collator(data):
    return_dict = dict()
    for key in data[0]:
        if key in ['input_ids', 'attention_mask', 'labels']:
            return_dict[key] = torch.stack([d[key] for d in data]).to(self.model.device)
    return return_dict