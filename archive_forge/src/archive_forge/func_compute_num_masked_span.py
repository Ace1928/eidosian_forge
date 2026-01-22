import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
def compute_num_masked_span(input_length):
    """Given input length, compute how many spans should be masked"""
    num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
    num_masked_span = max(num_masked_span, min_masks)
    if num_masked_span * mask_length > sequence_length:
        num_masked_span = sequence_length // mask_length
    if input_length - (mask_length - 1) < num_masked_span:
        num_masked_span = max(input_length - (mask_length - 1), 0)
    return num_masked_span