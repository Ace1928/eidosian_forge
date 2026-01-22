import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
    params = list(model.parameters()) if is_module else []
    in_vars, _ = _flatten((args, params))
    compiled_fn = model
    if force_trace:
        compiled_fn.clear_cache()
    if assert_compiled:
        hits = compiled_fn.hits
    out = model(*args)
    if assert_compiled and compiled_fn.hits == hits:
        raise RuntimeError('failed to use the compiled function')
    if not isinstance(out, tuple):
        out = (out,)
    if loss_fn == torch.sum and len(out) != 1:
        raise ValueError(f'Model returns {len(out)} outputs, but default loss function (torch.sum) can only handle a single output')
    out_vars, _ = _flatten(out)
    saved_outs = [v.detach().clone(memory_format=torch.preserve_format) for v in out_vars]
    loss = loss_fn(*out)
    grads = torch.autograd.grad([loss], in_vars)
    saved_grads = [v.detach().clone(memory_format=torch.preserve_format) for v in grads]
    return (saved_outs, saved_grads)