import collections
import functools
import itertools
import logging
import os
import random
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._logging
from torch._guards import compile_context, CompileContext, CompileId, tracing
from torch._utils_internal import log_compilation_event, signpost_event
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph_module import _forward_from_src as original_forward_from_src
from torch.utils._traceback import format_traceback_short
from . import config, exc
from .allowed_functions import is_allowed, is_numpy
from .backends.registry import CompilerFn
from .bytecode_analysis import remove_dead_code, remove_pointless_jumps
from .bytecode_transformation import (
from .cache_size import (
from .eval_frame import always_optimize_code_objects, skip_code, TorchPatcher
from .exc import (
from .guards import (
from .hooks import Hooks
from .output_graph import OutputGraph
from .replay_record import ExecutionRecord
from .symbolic_convert import InstructionTranslator, SpeculationLog
from .types import BytecodeHook
from .utils import (
from collections import OrderedDict
from torch.utils.hooks import RemovableHandle
def convert_frame(compiler_fn: CompilerFn, hooks: Hooks):
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    inner_convert = convert_frame_assert(compiler_fn, one_graph=False)

    def _convert_frame(frame: types.FrameType, cache_entry, hooks: Hooks, frame_state):
        counters['frames']['total'] += 1
        try:
            result = inner_convert(frame, cache_entry, hooks, frame_state)
            counters['frames']['ok'] += 1
            return result
        except Exception as e:
            if isinstance(e, UncapturedHigherOrderOpError):
                raise
            soft_fail = isinstance(e, Unsupported)
            if not config.suppress_errors and (not soft_fail):
                raise
            record_filename = getattr(e, 'record_filename', None)
            code = frame.f_code
            error_msg = format_error_msg(e, code, record_filename, frame)
            if soft_fail:
                log.info(error_msg, exc_info=True)
            else:
                log.warning(error_msg, exc_info=True)
        return None
    _convert_frame._torchdynamo_orig_callable = compiler_fn
    _convert_frame._clone_with_backend = lambda backend: convert_frame(backend, hooks)
    return _convert_frame