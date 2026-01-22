from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
import torch
import torch.utils._device
from torch._dynamo.source import (
from torch._guards import (
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
def compile_check_fn(self, builder, guards_out, guard_fail_fn):
    largs = builder.argnames
    largs += ['**___kwargs_ignored']
    guards_log.debug('GUARDS:')
    code_parts = ['___guarded_code.valid', '___check_global_state()']
    verbose_code_parts = code_parts[:]

    def add_code_part(code, guard, log_only=False):
        extra = ''
        if guard.user_stack:
            for fs in reversed(guard.user_stack):
                if fs.filename not in uninteresting_files():
                    extra = f'  # {format_frame(fs, line=True)}'
                    break
        elif guard.stack:
            extra = f'  # {format_frame(guard.stack.summary()[-1])}'
        guards_log.debug('%s', f'{code:<60}{extra}')
        if verbose_guards_log.isEnabledFor(logging.DEBUG):
            maybe_stack = ''
            maybe_user_stack = ''
            if guard is not None:
                if guard.stack:
                    maybe_stack = f'\nStack:\n{''.join(guard.stack.format())}'
                if guard.user_stack:
                    maybe_user_stack = f'\nUser stack:\n{''.join(guard.user_stack.format())}'
            verbose_guards_log.debug('Guard: %s%s%s', code, maybe_stack, maybe_user_stack)
        if not log_only:
            code_parts.append(code)
            verbose_code_parts.append(f'{code:<60}{extra}')
    seen = set()
    for gcl in builder.code:
        for code in gcl.code_list:
            if code not in seen:
                add_code_part(code, gcl.guard)
                seen.add(code)
    tensor_check_names = builder.tensor_check_names
    check_tensors_fn = None
    check_tensors_verbose_fn = None
    if tensor_check_names:
        assert not self.output_graph.export, 'Illegal to set tensor_check_names in export.'
        tensor_check_examples = builder.tensor_check_examples

        def convert(size_or_stride):
            converted: List[Optional[int]] = []
            for dim in size_or_stride:
                if not is_symbolic(dim):
                    converted.append(dim)
                else:
                    assert isinstance(dim, torch.SymInt)
                    converted.append(dim.node.maybe_as_int())
            return converted
        dynamic_dims_sizes = [convert(self.output_graph.tensor_weakref_to_sizes_strides[t]['size']) for t in tensor_check_examples]
        dynamic_dims_strides = [convert(self.output_graph.tensor_weakref_to_sizes_strides[t]['stride']) for t in tensor_check_examples]
        tensor_guards = TensorGuards(*tensor_check_examples, dynamic_dims_sizes=dynamic_dims_sizes, dynamic_dims_strides=dynamic_dims_strides)
        check_tensors_fn = tensor_guards.check
        check_tensors_verbose_fn = tensor_guards.check_verbose
        tensor_check_args = ', '.join(tensor_check_names + ['tensor_check_names=tensor_check_names'])
        code_parts.append(f'___check_tensors({tensor_check_args})')
        verbose_code_parts.append(f'___check_tensors({tensor_check_args})')
        tensor_check_guards = builder.tensor_check_guards
        for i, name in enumerate(tensor_check_names):
            t = tensor_check_examples[i]
            pytype = type(t)
            dispatch_key = (torch._C._dispatch_keys(t) | torch._C._dispatch_tls_local_include_set()) - torch._C._dispatch_tls_local_exclude_set()
            dtype = t.dtype
            device_index = t.device.index
            requires_grad = t.requires_grad
            sizes = dynamic_dims_sizes[i]
            strides = dynamic_dims_strides[i]
            add_code_part(f'check_tensor({name}, {pytype.__qualname__}, {dispatch_key}, {dtype}, device={device_index}, requires_grad={requires_grad}, size={sizes}, stride={strides})', tensor_check_guards[i], log_only=True)
    aotautograd_guards: List[GuardEnvExpr] = self.output_graph.tracing_context.guards_context.aotautograd_guards if self.output_graph else []
    for guard in aotautograd_guards:
        if isinstance(guard, DuplicateInputs):
            source_a = guard.input_source_a
            source_b = guard.input_source_b
            add_code_part(f'{source_a.name()} is {source_b.name()}', None)
        else:
            raise RuntimeError(f'Unknown GuardEnvExpr: {guard}')
    for gcl in builder.shape_env_code:
        for code in gcl.code_list:
            add_code_part(code, gcl.guard)
    global_state = convert_frame.initial_global_state
    if global_state is None:
        global_state = convert_frame.GlobalStateGuard()
    closure_vars = {'___guarded_code': self, '___check_tensors': check_tensors_fn, '___check_tensors_verbose': check_tensors_verbose_fn, '___check_global_state': global_state.check, 'tensor_check_names': tensor_check_names, **SYMPY_INTERP, **CLOSURE_VARS}
    unique_code_parts = list(unique(code_parts))
    make_guard_fn_args = ', '.join(closure_vars.keys())
    guard_body, pycode = build_guard_function(unique_code_parts, make_guard_fn_args)
    if os.environ.get('TORCHDYNAMO_PRINT_GUARDS', None) == '1':
        print('GUARDS\n', guard_body)
    out: Dict[str, Any] = dict()
    exec(pycode, builder.scope, out)
    guard_fn = out['___make_guard_fn'](*closure_vars.values())
    guard_fn.closure_vars = closure_vars
    guard_fn.args = largs
    guard_fn.code_parts = code_parts
    guard_fn.verbose_code_parts = verbose_code_parts
    guard_fn.global_scope = {'G': builder.scope['G']}
    guard_fn.guard_fail_fn = guard_fail_fn
    return guard_fn