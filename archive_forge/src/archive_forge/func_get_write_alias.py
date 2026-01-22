import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
def get_write_alias(x):
    if len(x.alias_set) == 0:
        return None
    alias_set = list(x.alias_set)
    assert len(alias_set) == 1
    if x.is_write:
        return alias_set[0]
    return None