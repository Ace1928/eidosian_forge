import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
@compatibility(is_backward_compatible=False)
def type_matches(signature_type: Any, argument_type: Any):
    sig_origin_type = getattr(signature_type, '__origin__', signature_type)
    if signature_type is argument_type:
        return True
    if sig_origin_type is typing.Union and signature_type != argument_type:
        sig_contained = signature_type.__args__
        return any((type_matches(c, argument_type) for c in sig_contained))
    if signature_type is List[int] and argument_type is int:
        return True
    if getattr(signature_type, '__origin__', None) in {list, List}:
        sig_el_type = signature_type.__args__[0]
        if not inspect.isclass(sig_el_type):
            warnings.warn(f'Does not support nested parametric types, got {signature_type}. Please file a bug.')
            return False
        if getattr(argument_type, '__origin__', None) in {list, List}:
            return issubclass(argument_type.__args__[0], sig_el_type)

        def is_homogeneous_tuple(t):
            if getattr(t, '__origin__', None) not in {tuple, Tuple}:
                return False
            contained = t.__args__
            if t.__args__ == ((),):
                return True
            return all((c is Ellipsis or issubclass(c, sig_el_type) for c in contained))
        return is_homogeneous_tuple(argument_type)
    if signature_type is int and argument_type is torch.dtype:
        return True
    if signature_type is numbers.Number and argument_type in {int, float}:
        return True
    if inspect.isclass(argument_type) and inspect.isclass(signature_type):
        return issubclass(argument_type, signature_type)
    return False