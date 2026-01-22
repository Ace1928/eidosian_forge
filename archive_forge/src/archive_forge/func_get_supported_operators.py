from __future__ import annotations
import copy
from typing import List, Set
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
@classmethod
def get_supported_operators(cls) -> List[OperatorConfig]:
    return [get_embedding_operators_config()]