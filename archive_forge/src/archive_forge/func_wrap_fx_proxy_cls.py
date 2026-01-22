import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import operator
import re
import sys
import types
from typing import List, NamedTuple, Optional, Union
import torch
from torch import SymInt
from torch._guards import GuardSource, TracingContext
from torch._ops import HigherOrderOperator
from torch._streambase import _EventBase, _StreamBase
from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.immutable_collections import immutable_list
from torch.nested._internal.nested_tensor import NestedTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import TensorWeakRef
from .. import config, mutation_guard, replay_record, skipfiles, trace_rules
from ..allowed_functions import (
from ..device_interface import get_registered_device_interfaces
from ..exc import InternalTorchDynamoError, unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..side_effects import SideEffects
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
from .dicts import (
from .distributed import (
from .functions import (
from .higher_order_ops import TorchHigherOrderOperatorVariable
from .lazy import LazyVariableTracker
from .lists import (
from .misc import (
from .nn_module import FSDPManagedNNModuleVariable, UnspecializedNNModuleVariable
from .optimizer import OptimizerVariable
from .tensor import (
from .torch import torch_special_class_types, TorchVariable
from .torch_function import build_torch_function_fn, TensorWithTFOverrideVariable
from .user_defined import (
def wrap_fx_proxy_cls(target_cls, tx, proxy, example_value=None, subclass_type=None, **options):
    from ..symbolic_convert import InstructionTranslatorBase
    assert isinstance(tx, InstructionTranslatorBase)
    if 'guards' in options and options['guards'] is not None:
        tx.output.guards.update(options['guards'])
    assert 'example_value' not in proxy.node.meta, f'{proxy.node.meta['example_value']}'
    initial_example_value = example_value

    def _clone_input(value):
        if isinstance(value, torch.Tensor):
            if not (isinstance(value, FakeTensor) or (torch._is_functional_tensor(value) and maybe_get_fake_mode(value) is tx.fake_mode) or value.is_nested):
                value = clone_input(value)
        return value
    with preserve_rng_state():
        if example_value is None:
            example_value = get_fake_value(proxy.node, tx, allow_non_graph_fake=True)
        elif maybe_get_fake_mode(example_value) is tx.fake_mode:
            pass
        elif isinstance(example_value, torch.Tensor):
            if tx.export:
                with torch._C.DisableTorchFunctionSubclass():
                    proxy.tracer.real_value_cache[proxy.node] = _clone_input(example_value)
            kwargs = {'is_tensor': target_cls in (TensorVariable, TensorWithTFOverrideVariable)}
            assert 'source' in options and options['source'] is not None
            kwargs['source'] = options['source']
            example_value = wrap_to_fake_tensor_and_record(example_value, tx=tx, **kwargs)
        if isinstance(example_value, torch.Tensor) and maybe_get_fake_mode(example_value) is not tx.fake_mode:
            raise InternalTorchDynamoError(f'`example_value` needs to be a `FakeTensor`wrapped by this instance of Dynamo. Found: {example_value}')
    if isinstance(example_value, torch.Tensor):
        is_parameter = isinstance(example_value, torch.nn.Parameter)
        should_specialize = options.pop('should_specialize', False)
        if is_parameter or should_specialize:
            specialized_value = initial_example_value
        else:
            specialized_value = None
        example_value = _clone_input(example_value)
        proxy.node.meta['example_value'] = example_value
        specialized_props = target_cls.specialize(example_value)
        if isinstance(example_value, torch._subclasses.fake_tensor.FakeTensor) and example_value.fake_mode is tx.fake_mode:
            tensor_type = subclass_type if subclass_type else torch.Tensor
            specialized_props['class_type'] = torch.nn.Parameter if is_parameter else tensor_type
        specialized_props['specialized_value'] = specialized_value
        options.update(specialized_props)
        return target_cls(proxy, **options)
    elif hasattr(proxy.node.target, '__name__') and proxy.node.target.__name__ == 'set_state' and isinstance(proxy.node.target.__self__, torch._C.Generator) or proxy.node.target == torch.random.set_rng_state:
        return TorchVariable(proxy.node.target)
    elif proxy.node.target == torch._C._DisableFuncTorch or proxy.node.target == torch.cuda._is_in_bad_fork:
        return UserDefinedObjectVariable(example_value)
    elif istype(example_value, torch.Size) and all((isinstance(x, int) for x in example_value)):
        sizes = [ConstantVariable.create(x) for x in example_value]
        return SizeVariable(sizes, **options)
    elif isinstance(example_value, (tuple, list, set)):
        proxy.node.meta['example_value'] = example_value
        unpacked = []
        for i, val in enumerate(example_value):
            if val is None:
                unpacked.append(ConstantVariable.create(None, **options))
            else:
                unpacked.append(wrap_fx_proxy_cls(target_cls, tx, proxy.tracer.create_proxy('call_function', operator.getitem, (proxy, i), {}), example_value=val, **options))
        if isinstance(example_value, torch.Size):
            return SizeVariable(unpacked, proxy, **options)
        elif istype(example_value, tuple):
            return TupleVariable(unpacked, **options)
        elif istype(example_value, (list, immutable_list)):
            return ListVariable(unpacked, mutable_local=MutableLocal(), **options)
        elif istype(example_value, set):
            return SetVariable(unpacked, mutable_local=MutableLocal(), **options)
        else:
            assert example_value.__class__.__module__ == 'torch.return_types' or hasattr(example_value, '_fields'), f'expected {example_value.__class__.__module__} == torch.return_types or named tuple but got {type(example_value)}'
            return NamedTupleVariable(unpacked, example_value.__class__, **options)
    elif example_value is None or proxy.node.target is torch.manual_seed:
        return ConstantVariable.create(None, **options)
    elif isinstance(example_value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        proxy.node.meta['example_value'] = example_value
        return SymNodeVariable(proxy, example_value, **options)
    elif inspect.isclass(proxy.node.target) and issubclass(proxy.node.target, _StreamBase) or proxy.node.target in [device_interface.current_stream for _, device_interface in get_registered_device_interfaces()]:
        proxy.node.meta['example_value'] = example_value
        return StreamVariable(proxy, example_value, example_value.device.type, **options)
    elif inspect.isclass(proxy.node.target) and issubclass(proxy.node.target, _EventBase) or proxy.node.target in [device_interface.Event for _, device_interface in get_registered_device_interfaces()]:
        proxy.node.meta['example_value'] = example_value
        return EventVariable(proxy, example_value, **options)
    elif proxy.node.target == 'query' and proxy.node.op == 'call_method':
        proxy.node.meta['example_value'] = example_value
        return ConstantVariable(example_value, **options)
    elif example_value is not None and isinstance(example_value, _EventBase) and (proxy.node.target == 'record_event') and (proxy.node.op == 'call_method'):
        proxy.node.meta['example_value'] = example_value
        return EventVariable(proxy, example_value, **options)
    elif isinstance(example_value, int) and proxy.node.target in [torch.sym_int, getattr, operator.getitem, torch._utils._element_size, torch.seed, operator.mod, getattr(torch.distributed, 'get_rank', _missing), getattr(torch.distributed, 'get_world_size', _missing), torch._constrain_as_value, torch._constrain_as_size]:
        proxy.node.meta['example_value'] = example_value
        return ConstantVariable.create(example_value, **options)
    else:
        unimplemented('torch.* op returned non-Tensor ' + f'{typestr(example_value)} {proxy.node.op} {proxy.node.target}')