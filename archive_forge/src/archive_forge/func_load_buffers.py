import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def load_buffers(mod: nn.Module, names: Sequence[str], buffers: Sequence[Tensor], as_params: bool=False) -> None:
    accessor = NamedMemberAccessor(mod)
    accessor.set_tensors(names, buffers)