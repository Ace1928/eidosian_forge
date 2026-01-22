import collections.abc
import copy
import dataclasses
import inspect
import sys
import types
import warnings
from typing import (
from typing_extensions import Annotated, Self, get_args, get_origin, get_type_hints
from . import _fields, _unsafe_cache
from ._typing import TypeForm
def resolve_generic_types(cls: TypeOrCallable) -> Tuple[TypeOrCallable, Dict[TypeVar, TypeForm[Any]]]:
    """If the input is a class: no-op. If it's a generic alias: returns the origin
    class, and a mapping from typevars to concrete types."""
    annotations: Tuple[Any, ...] = ()
    if get_origin(cls) is Annotated:
        cls, annotations = unwrap_annotated(cls)
    origin_cls = get_origin(unwrap_newtype(cls)[0])
    type_from_typevar: Dict[TypeVar, TypeForm[Any]] = {}
    if hasattr(cls, '__self__'):
        self_type = getattr(cls, '__self__')
        if inspect.isclass(self_type):
            type_from_typevar[cast(TypeVar, Self)] = self_type
        else:
            type_from_typevar[cast(TypeVar, Self)] = self_type.__class__
    if origin_cls is not None and hasattr(origin_cls, '__parameters__') and hasattr(origin_cls.__parameters__, '__len__'):
        typevars = origin_cls.__parameters__
        typevar_values = get_args(unwrap_newtype(cls)[0])
        assert len(typevars) == len(typevar_values)
        cls = origin_cls
        type_from_typevar.update(dict(zip(typevars, typevar_values)))
    if hasattr(cls, '__orig_bases__'):
        bases = getattr(cls, '__orig_bases__')
        for base in bases:
            origin_base = unwrap_origin_strip_extras(base)
            if origin_base is base or not hasattr(origin_base, '__parameters__'):
                continue
            typevars = origin_base.__parameters__
            typevar_values = get_args(base)
            type_from_typevar.update(dict(zip(typevars, typevar_values)))
    if len(annotations) == 0:
        return (cls, type_from_typevar)
    else:
        return (Annotated.__class_getitem__((cls, *annotations)), type_from_typevar)