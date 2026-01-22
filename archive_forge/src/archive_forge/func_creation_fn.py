import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
def creation_fn(self, all_args, *, is_runtime: bool):
    curr_args = all_args[self.flat_tensor_start_idx:self.flat_tensor_start_idx + self.arg_count]
    assert len(curr_args) == len(self.inner_keys), f'inner_keys: {str(self.inner_keys)}. len(curr_args): {len(curr_args)}'
    out = type(self.original_subclass).__tensor_unflatten__(dict(zip(self.inner_keys, curr_args)), self.meta)
    if not is_runtime:
        torch._mirror_autograd_meta_to(self.original_subclass, out)
    return out