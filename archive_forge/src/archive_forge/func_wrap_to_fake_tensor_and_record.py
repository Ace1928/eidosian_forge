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
def wrap_to_fake_tensor_and_record(e, tx, *, source: Optional[Source], is_tensor: bool):
    if type(e) in (torch.Tensor, torch.nn.Parameter, FakeTensor) or isinstance(e, torch.Tensor) or is_traceable_wrapper_subclass(e):
        assert source is not None
        static_shapes, reason = tensor_always_has_static_shape(e, is_tensor, guard_source=source.guard_source())
        symbolic_context = None
        if not e.is_nested:
            symbolic_context = _automatic_dynamic(e, tx, source, static_shapes)
        if symbolic_context:
            tx.output.tracing_context.tensor_to_context[e] = symbolic_context
        log.debug('wrap_to_fake %s %s %s %s', source.name(), tuple(e.shape), symbolic_context.dynamic_sizes if symbolic_context is not None else None, symbolic_context.constraint_sizes if symbolic_context is not None else None)
        fake_e = wrap_fake_exception(lambda: tx.fake_mode.from_tensor(e, source=source, symbolic_context=symbolic_context))
        if is_tensor and (not (static_shapes and source.is_nn_module())):
            tx.output.tracked_fakes.append(TrackedFake(fake_e, source, symbolic_context.constraint_sizes if symbolic_context is not None else None))
            tx.output.tracked_fakes_id_to_source[id(e)].append(source)
        tx.output.tensor_weakref_to_sizes_strides[e] = {'size': fake_e.size(), 'stride': fake_e.stride()}
        return fake_e
    else:
        return e