import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
def get_qkv_in_bmghk(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.query.ndim == 5:
        return (self.query, self.key, self.value)
    if self.query.ndim == 4:
        return (self.query.unsqueeze(2), self.key.unsqueeze(2), self.value.unsqueeze(2))
    if self.value.ndim == 3:
        return (self.query[:, :, None, None], self.key[:, :, None, None], self.value[:, :, None, None])
    assert False