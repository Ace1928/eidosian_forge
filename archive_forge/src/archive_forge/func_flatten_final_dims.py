from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload
import torch
import torch.nn as nn
import torch.types
def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    return t.reshape(t.shape[:-no_dims] + (-1,))