import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def sparse24_t(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, Sparse24Tensor)
    assert len(self.shape) == 2
    return self.__class__((self.shape[-1], self.shape[0]), packed=self.packed_t, meta=self.meta_t, packed_t=self.packed, meta_t=self.meta, threads_masks=self.threads_masks.transpose(0, 1))