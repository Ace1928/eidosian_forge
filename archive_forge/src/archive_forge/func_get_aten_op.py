from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def get_aten_op(fn: Callable, name: str):
    """
    Given the __module__ of reference and its name, it returns
    (our best guess of) the ATen name of the associated operation

    Note: In ATen, the __name__ of a function within a module often
    starts by the module name. E.g. linalg_eigh, or special_zeta
    """
    module = fn.__module__
    prefix = 'torch._refs'
    assert module.startswith(prefix)
    module = module[len(prefix):]
    if module:
        module = module[1:]
        module = module.replace('.', '_')
        module = module + '_'
    return getattr(torch._ops.ops.aten, f'{module}{name}')