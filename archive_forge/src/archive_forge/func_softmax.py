import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
def softmax(hidden_state, dim, onnx_trace=False):
    if onnx_trace:
        return nn.functional.softmax(hidden_state.float(), dim=dim)
    else:
        return nn.functional.softmax(hidden_state, dim=dim, dtype=torch.float32)