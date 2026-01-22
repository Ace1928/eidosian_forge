import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
def call_apply(self, tx, args, kwargs):
    requires_grad = False

    def visit(node):
        nonlocal requires_grad
        if isinstance(node, variables.TensorVariable):
            if node.requires_grad is not False:
                requires_grad = True
        if isinstance(node, variables.NNModuleVariable):
            if node.is_training(tx):
                requires_grad = True
        return node
    VariableTracker.apply(visit, (args, kwargs))
    ctx = AutogradFunctionContextVariable.create(tx)
    args = [ctx, *args]
    if requires_grad and torch.is_grad_enabled() and config.capture_autograd_function:
        if self.fn_cls.setup_context != torch.autograd.function._SingleLevelFunction.setup_context:
            unimplemented('NYI - autograd.Function with custom setup_context method')
        vjp_fn = self.fn_cls.vjp
        if vjp_fn is not torch.autograd.Function.vjp:
            unimplemented('NYI - User defind vjp')
        jvp_fn = self.fn_cls.jvp
        if jvp_fn is not torch.autograd.Function.jvp:
            unimplemented('NYI - User defind jvp')
        from .higher_order_ops import safe_or_raise_always_restore, TorchHigherOrderOperatorVariable
        trampoline_autograd_apply = produce_trampoline_autograd_apply(self.fn_cls)
        trampoline_autograd_fwd = produce_trampoline_autograd_fwd(self.fn_cls)
        trampoline_autograd_bwd = produce_trampoline_autograd_bwd(self.fn_cls)
        graph_checkpoint, checkpoint = (tx.output.graph, tx.copy_graphstate())
        module_source = AttrSource(tx.import_source(self.fn_cls.__module__), self.fn_cls.__name__)
        fwd_bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(tx.output, parent=tx.output.current_tracer, source_target='autograd.Function')
        higher_order_autograd_fn = TorchHigherOrderOperatorVariable.make(trampoline_autograd_fwd, source=AttrSource(module_source, 'forward'), fwd_bwd_tracer=fwd_bwd_tracer)
        speculated_fwd_result = higher_order_autograd_fn.call_function(tx, args, kwargs)
        if isinstance(speculated_fwd_result, variables.TupleVariable):
            bwd_args = [ctx, *speculated_fwd_result.items]
        else:
            bwd_args = [ctx, speculated_fwd_result]
        safe_or_raise_always_restore(tx, graph_checkpoint, checkpoint, TorchHigherOrderOperatorVariable.make(trampoline_autograd_bwd, source=AttrSource(module_source, 'backward'), fwd_bwd_tracer=fwd_bwd_tracer), bwd_args)
        args = args[1:]
        return TorchHigherOrderOperatorVariable.make(trampoline_autograd_apply, fwd_bwd_tracer=None).call_function(tx, args, kwargs)
    if self.source:
        source = AttrSource(AttrSource(self.source, '__class__'), 'forward')
    else:
        source = None
    fn = self.fn_cls.forward
    if isinstance(fn, types.FunctionType):
        return variables.UserFunctionVariable(fn, source=source).call_function(tx, args, kwargs)
    elif isinstance(fn, types.MethodType):
        return variables.UserMethodVariable(fn.__func__, variables.UserDefinedClassVariable(self.fn_cls), source=source).call_function(tx, args, kwargs)
    else:
        unimplemented(f'non-function or method in subclass of torch.autograd.Function: {fn}')