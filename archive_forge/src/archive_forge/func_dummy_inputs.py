import copy
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_marian import MarianConfig
@property
def dummy_inputs(self):
    pad_token = self.config.pad_token_id
    input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
    dummy_inputs = {'attention_mask': input_ids.ne(pad_token), 'input_ids': input_ids, 'decoder_input_ids': input_ids}
    return dummy_inputs