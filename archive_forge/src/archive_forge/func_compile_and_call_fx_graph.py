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
@torch._guards.TracingContext.clear_frame()
def compile_and_call_fx_graph(self, tx, rv, root):
    """
        Generate code from self.graph and return the Instruction()s to
        call that generated code.
        """
    from .decorators import disable
    assert self.should_exit
    name = unique_id('__compiled_fn')
    assert isinstance(rv, list)
    assert isinstance(root, FakeRootModule)
    self.create_node('output', 'output', (self.current_tracer.create_arg(tuple((x.as_proxy() for x in rv))),), {})
    self.insert_deferred_runtime_asserts(root, name)
    self.remove_unused_graphargs()
    ncalls = count_calls(self.graph)
    counters['stats']['calls_captured'] += ncalls
    self.real_value_cache.clear()
    gm = fx.GraphModule(root, self.graph)
    for register_finalizer in self.register_finalizer_fns:
        register_finalizer(gm)
    gm.compile_subgraph_reason = self.compile_subgraph_reason
    graph_code_log.debug('%s', lazy_format_graph_code(name, gm))
    graph_tabular_log.debug('%s', lazy_format_graph_tabular(name, gm))
    graph_sizes_log.debug('%s', LazyString(lambda: self.get_graph_sizes_log_str(name)))
    self.call_cleanup_hooks()
    old_fake_mode = self.tracing_context.fake_mode
    if not self.export:
        backend_fake_mode = torch._subclasses.FakeTensorMode(shape_env=old_fake_mode.shape_env)
        self.tracing_context.fake_mode = backend_fake_mode
    with self.restore_global_state():
        compiled_fn = self.call_user_compiler(gm)
    compiled_fn = disable(compiled_fn)
    counters['stats']['unique_graphs'] += 1
    self.install_global(name, compiled_fn)
    cg = PyCodegen(tx)
    cg.make_call_generated_code(name)
    return cg.get_instructions()