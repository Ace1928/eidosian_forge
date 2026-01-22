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
@property
def decoder_cross_attentions(self):
    warnings.warn('`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions` instead.', FutureWarning)
    return self.cross_attentions