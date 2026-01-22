from typing import Tuple
import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403
def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)