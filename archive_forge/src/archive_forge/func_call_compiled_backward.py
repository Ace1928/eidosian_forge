import logging
from contextlib import nullcontext
from functools import wraps
from typing import Any, List, Optional
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._logging import getArtifactLogger
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from .. import config
from .dispatch_and_compile_graph import (
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .runtime_wrappers import (
from .schemas import (
from .subclass_utils import unwrap_tensor_subclasses, wrap_tensor_subclasses
from .utils import (
def call_compiled_backward():
    if ctx._is_compiled_autograd_tracing():
        symints = ctx._get_compiled_autograd_symints()
        assert len(symints) == len(ctx.symints)
        all_args[:len(symints)] = symints
        context = torch._C._DisableAutocast if disable_amp else nullcontext
        with context():
            out = normalize_as_list(bw_module(*all_args))
        out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
        return tuple(out)
    ctx.maybe_clear_saved_tensors()
    if CompiledFunction.compiled_bw is None:
        context = torch._C._DisableAutocast if disable_amp else nullcontext
        with tracing(saved_context), context(), track_graph_compiling(aot_config, 'backward'):
            CompiledFunction.compiled_bw = aot_config.bw_compiler(bw_module, placeholder_list)
    out = call_func_at_runtime_with_args(CompiledFunction.compiled_bw, all_args, steal_args=True, disable_amp=disable_amp)
    out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
    return tuple(out)