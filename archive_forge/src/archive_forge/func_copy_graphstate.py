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
def copy_graphstate(self) -> OutputGraphState:
    """Create a checkpoint of the current state by copying everything"""
    assert self.param_name_to_source is not None
    guards_graph_state = self.tracing_context.guards_context.copy_graphstate()
    module_state = self.tracing_context.module_context.copy_graphstate()
    global_state = self.tracing_context.global_context.copy_graphstate()
    state = OutputGraphState(dict(self.input_source_to_var), list(self.tracked_fakes), guards_graph_state, module_state, list(self.register_finalizer_fns), global_state, dict(self.param_name_to_source), self.side_effects.clone(), self.timestamp, set(self.non_compliant_ops), set(self.compliant_custom_ops))
    self.timestamp += 1
    return state