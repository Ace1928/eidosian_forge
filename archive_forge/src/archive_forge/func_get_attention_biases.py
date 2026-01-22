import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
def get_attention_biases(self, device):
    if self.training:
        return self.attention_biases[:, self.attention_bias_idxs]
    else:
        device_key = str(device)
        if device_key not in self.attention_bias_cache:
            self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
        return self.attention_bias_cache[device_key]