from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
def emit_feature_change(values: T.Dict[_T, T.Union[str, T.Tuple[str, str]]], feature: T.Union[T.Type['FeatureDeprecated'], T.Type['FeatureNew']]) -> None:
    for n, version in values.items():
        if isinstance(version, tuple):
            version, msg = version
        else:
            msg = None
        warning: T.Optional[str] = None
        if isinstance(n, ContainerTypeInfo):
            if n.check_any(value):
                warning = f'of type {n.description()}'
        elif isinstance(n, type):
            if isinstance(value, n):
                warning = f'of type {n.__name__}'
        elif isinstance(value, list):
            if n in value:
                warning = f'value "{n}" in list'
        elif isinstance(value, dict):
            if n in value.keys():
                warning = f'value "{n}" in dict keys'
        elif n == value:
            warning = f'value "{n}"'
        if warning:
            feature.single_use(f'"{name}" keyword argument "{info.name}" {warning}', version, subproject, msg, location=node)