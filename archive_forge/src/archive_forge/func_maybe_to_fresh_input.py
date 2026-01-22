import dataclasses
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import py_sym_types
def maybe_to_fresh_input(idx, t, meta):
    if not isinstance(t, torch.Tensor):
        return t
    if idx in meta.mutated_inp_runtime_indices:
        mutated_inp_idx = meta.mutated_inp_runtime_indices.index(idx)
        if meta.input_info[idx].requires_grad and meta.input_info[idx].mutates_data:
            return t.clone()
        if meta.input_info[idx] and meta.input_info[idx].mutates_metadata:
            return t.view(t.shape)
    return t