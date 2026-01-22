import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
@allow_in_graph
def sparsify24(x: torch.Tensor, algo: str='', gradient: str=GRADIENT_SP24, backend: str=BACKEND_CUTLASS) -> Sparse24Tensor:
    return _Sparsify24Func.apply(x, algo, gradient, backend)