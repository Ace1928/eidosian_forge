import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_nllb_moe import NllbMoeConfig
def normalize_router_probabilities(self, router_probs, top_1_mask, top_2_mask):
    top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
    top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
    denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
    top_1_max_probs = top_1_max_probs / denom_s
    top_2_max_probs = top_2_max_probs / denom_s
    return (top_1_max_probs, top_2_max_probs)