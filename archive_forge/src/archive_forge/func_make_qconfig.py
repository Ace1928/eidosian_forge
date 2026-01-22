import copy
import operator
import torch
from typing import Any, Callable, Optional, Tuple
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization.quantize_fx import (
def make_qconfig(obs_ctr: _PartialWrapper) -> QConfig:
    """
        Make a QConfig with fixed qparams observers or fake quantizes.
        """
    if isinstance(obs_ctr(), FakeQuantizeBase):
        weight = default_weight_fake_quant
    else:
        weight = default_weight_observer
    return QConfig(activation=obs_ctr, weight=weight)