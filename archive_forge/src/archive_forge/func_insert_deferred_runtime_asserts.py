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
def insert_deferred_runtime_asserts(self, root, name) -> None:
    """
        During tracing, we may have discovered that some data-dependent values
        had runtime assert on them; e.g., torch.empty(x.item()) induces a runtime
        that x.item() >= 0.  This asserts can happen unpredictably during fake
        tensor propagation, so we cannot conveniently insert them into the FX graph
        when they occur.  Instead, we accumulate them in the ShapeEnv, and in this
        pass insert them into the graph as proper tests.
        """
    ras_by_symbol = self.shape_env.deferred_runtime_asserts.copy()
    if not any((ras for ras in ras_by_symbol.values())):
        return
    gm = fx.GraphModule(root, self.graph)
    graph_code_log.debug('%s', lazy_format_graph_code(f'pre insert_deferred_runtime_asserts {name}', gm))
    symbol_to_proxy = {}
    placeholders = set()
    last_placeholder = None
    for node in self.graph.nodes:
        if node.op != 'placeholder':
            last_placeholder = node
            break
        placeholders.add(node)
    assert last_placeholder is not None
    needed_symbols: Set[sympy.Symbol] = set()
    for ras in ras_by_symbol.values():
        for ra in ras:
            needed_symbols.update(free_symbols(ra.expr))
    log.debug('needed_symbols = %s', needed_symbols)
    for node in self.graph.nodes:
        with self.graph.inserting_before(node.next if node not in placeholders else last_placeholder.next):
            if 'example_value' not in node.meta:
                continue
            defs = []

            def match_symbol(symint, cb):
                if isinstance(symint, torch.SymInt) and isinstance(symint.node, SymNode) and isinstance((s := symint.node.expr), sympy.Symbol) and (s not in symbol_to_proxy) and (s in needed_symbols):
                    symbol_to_proxy[s] = fx.Proxy(cb())
                    log.debug('symbol_to_proxy[%s] = %s', s, symbol_to_proxy[s])
                    defs.append(s)
            match_symbol(node.meta['example_value'], lambda: node)
            if isinstance((t := node.meta['example_value']), torch.Tensor):
                for i, s in enumerate(t.size()):
                    match_symbol(s, lambda: self.graph.call_method('size', (node, i)))
                for i, s in enumerate(t.stride()):
                    match_symbol(s, lambda: self.graph.call_method('stride', (node, i)))
                match_symbol(t.storage_offset(), lambda: self.graph.call_method('storage_offset', (node,)))
            for i0 in defs:
                ras = ras_by_symbol.pop(i0, [])
                for ra in ras:
                    log.debug('inserting runtime assert %s', ra.expr)
                    fvs = free_symbols(ra.expr)
                    missing = fvs - symbol_to_proxy.keys()
                    if missing:
                        i1 = sorted(missing)[0]
                        assert self.shape_env.is_unbacked_symint(i1), i1
                        ras_by_symbol.setdefault(i1, []).append(ra)
                    else:
                        res = sympy_interp(PythonReferenceAnalysis, symbol_to_proxy, ra.expr).node
                        res2 = self.graph.call_function(torch.ops.aten.scalar_tensor.default, (res,))
                        self.graph.call_function(torch.ops.aten._assert_async.msg, (res2, f'Deferred runtime assertion failed {ra.expr}'))