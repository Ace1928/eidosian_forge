from __future__ import annotations
import collections.abc
import inspect
import sys
import types
import typing
import warnings
from enum import Enum
from inspect import Parameter, isclass, isfunction
from io import BufferedIOBase, IOBase, RawIOBase, TextIOBase
from textwrap import indent
from typing import (
from unittest.mock import Mock
from ._config import ForwardRefPolicy
from ._exceptions import TypeCheckError, TypeHintWarning
from ._memo import TypeCheckMemo
from ._utils import evaluate_forwardref, get_stacklevel, get_type_name, qualified_name
def check_type_internal(value: Any, annotation: Any, memo: TypeCheckMemo) -> None:
    """
    Check that the given object is compatible with the given type annotation.

    This function should only be used by type checker callables. Applications should use
    :func:`~.check_type` instead.

    :param value: the value to check
    :param annotation: the type annotation to check against
    :param memo: a memo object containing configuration and information necessary for
        looking up forward references
    """
    if isinstance(annotation, ForwardRef):
        try:
            annotation = evaluate_forwardref(annotation, memo)
        except NameError:
            if memo.config.forward_ref_policy is ForwardRefPolicy.ERROR:
                raise
            elif memo.config.forward_ref_policy is ForwardRefPolicy.WARN:
                warnings.warn(f'Cannot resolve forward reference {annotation.__forward_arg__!r}', TypeHintWarning, stacklevel=get_stacklevel())
            return
    if annotation is Any or annotation is SubclassableAny or isinstance(value, Mock):
        return
    if not isclass(value) and SubclassableAny in type(value).__bases__:
        return
    extras: tuple[Any, ...]
    origin_type = get_origin(annotation)
    if origin_type is Annotated:
        annotation, *extras_ = get_args(annotation)
        extras = tuple(extras_)
        origin_type = get_origin(annotation)
    else:
        extras = ()
    if origin_type is not None:
        args = get_args(annotation)
        if origin_type in (tuple, Tuple) and annotation is not Tuple and (not args):
            args = ((),)
    else:
        origin_type = annotation
        args = ()
    for lookup_func in checker_lookup_functions:
        checker = lookup_func(origin_type, args, extras)
        if checker:
            checker(value, origin_type, args, memo)
            return
    if isclass(origin_type):
        if not isinstance(value, origin_type):
            raise TypeCheckError(f'is not an instance of {qualified_name(origin_type)}')
    elif type(origin_type) is str:
        warnings.warn(f'Skipping type check against {origin_type!r}; this looks like a string-form forward reference imported from another module', TypeHintWarning, stacklevel=get_stacklevel())