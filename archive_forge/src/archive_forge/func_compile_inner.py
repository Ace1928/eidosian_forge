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
@dynamo_timed(phase_name='entire_frame_compile')
def compile_inner(code: types.CodeType, one_graph: bool, hooks: Hooks, transform: Callable[[List[Instruction], Dict[str, Any]], Any]) -> Optional[GuardedCode]:
    nonlocal output
    for attempt in itertools.count():
        CompileContext.get().attempt = attempt
        try:
            out_code = transform_code_object(code, transform)
            break
        except exc.RestartAnalysis as e:
            log.info('Restarting analysis due to %s', LazyString(format_traceback_short, e.__traceback__))
            if attempt > 100:
                unimplemented('100+ RestartAnalysis() calls')
        except exc.SkipFrame as e:
            log.debug('Skipping frame %s %s                     %s %s', e, code.co_name, code.co_filename, code.co_firstlineno)
            if one_graph:
                log.debug('No graph captured with one_graph=True')
            return None

    def log_bytecode(prefix, name, filename, line_no, code):
        if bytecode_log.isEnabledFor(logging.DEBUG):
            bytecode_log.debug(format_bytecode(prefix, name, filename, line_no, code))
    log_bytecode('ORIGINAL BYTECODE', code.co_name, code.co_filename, code.co_firstlineno, code)
    log_bytecode('MODIFIED BYTECODE', code.co_name, code.co_filename, code.co_firstlineno, out_code)
    for hook in _bytecode_hooks.values():
        hook_output = hook(code, out_code)
        if hook_output is not None:
            out_code = hook_output
    orig_code_map[out_code] = code
    output_codes.add(out_code)
    assert output is not None
    if output.export and output.is_empty_graph():
        return None
    assert output.guards is not None
    CleanupManager.instance[out_code] = output.cleanups
    check_fn = CheckFunctionManager(output, hooks.guard_fail_fn if hooks else None)
    guarded_code = GuardedCode(out_code, check_fn.check_fn)
    if not output.is_empty_graph() and hooks.guard_export_fn is not None:
        hooks.guard_export_fn(output.guards)
    output.local_scope.clear()
    return guarded_code