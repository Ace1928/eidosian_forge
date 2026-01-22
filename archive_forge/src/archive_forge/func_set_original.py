import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
def set_original(self, value: Tensor) -> None:
    if torch.jit.is_scripting():
        raise RuntimeError('Parametrization is not working with scripting.')
    self.parametrizations[tensor_name].right_inverse(value)