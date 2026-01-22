import collections
import contextlib
import copy
import functools
import itertools
import logging
import operator
import re
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import sympy
import torch._guards
import torch._logging
import torch.nn
import torch.utils._pytree as pytree
from torch import fx
from torch._guards import (
from torch._utils_internal import signpost_event
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import free_symbols, is_symbolic, ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.reference import PythonReferenceAnalysis
from torch.utils.weak import WeakTensorKeyDictionary
from . import config, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import enter_new_scope
from .exc import (
from .guards import GuardBuilder, install_guard
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import (
from .utils import (
from .variables.base import VariableTracker
from .variables.builder import GraphArg, TrackedFake, VariableBuilder, wrap_fx_proxy
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch_function import TensorWithTFOverrideVariable
@dynamo_timed(phase_name='backend_compile')
def call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:
    assert self.compiler_fn is not None
    tot = 0
    placeholders = []
    for node in gm.graph.nodes:
        if node.op in ('call_function', 'call_method', 'call_module'):
            tot += 1
        if node.op == 'placeholder':
            placeholders.append(node)
    increment_op_count(tot)
    for pl in placeholders:
        arg = pl.meta['grapharg']
        pl._dynamo_source = arg.source
    gm._param_name_to_source = self.param_name_to_source
    gm._source_to_user_stacks = self.source_to_user_stacks
    try:
        name = self.compiler_fn.__name__ if hasattr(self.compiler_fn, '__name__') else ''
        _step_logger()(logging.INFO, f'calling compiler function {name}')
        compiler_fn = self.compiler_fn
        if config.verify_correctness:
            compiler_fn = WrapperBackend(compiler_fn)
        compiled_fn = compiler_fn(gm, self.example_inputs())
        _step_logger()(logging.INFO, f'done compiler function {name}')
        assert callable(compiled_fn), 'compiler_fn did not return callable'
    except exceptions_allowed_to_be_fallback as e:
        if self.has_user_defined_allowed_in_graph:
            raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(e.__traceback__) from None
        msg = f'Backend compiler failed with a fake tensor exception at \n{self.root_tx.format_frame_summary()}Adding a graph break.'
        unimplemented_with_warning(e, self.root_tx.f_code, msg)
    except SkipFrame as e:
        raise e
    except Exception as e:
        raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(e.__traceback__) from None
    signpost_event('dynamo', 'OutputGraph.call_user_compiler', {**self.co_fields, 'op_count': tot, 'node_count': len(gm.graph.nodes), 'input_count': len(placeholders)})
    return compiled_fn