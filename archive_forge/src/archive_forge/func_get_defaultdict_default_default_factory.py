from __future__ import annotations as _annotations
import collections
import collections.abc
import dataclasses
import decimal
import inspect
import os
import typing
from enum import Enum
from functools import partial
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any, Callable, Iterable, TypeVar
import typing_extensions
from pydantic_core import (
from typing_extensions import get_args, get_origin
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo
from pydantic.types import Strict
from ..config import ConfigDict
from ..json_schema import JsonSchemaValue, update_json_schema
from . import _known_annotated_metadata, _typing_extra, _validators
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._schema_generation_shared import GetCoreSchemaHandler, GetJsonSchemaHandler
def get_defaultdict_default_default_factory(values_source_type: Any) -> Callable[[], Any]:

    def infer_default() -> Callable[[], Any]:
        allowed_default_types: dict[Any, Any] = {typing.Tuple: tuple, tuple: tuple, collections.abc.Sequence: tuple, collections.abc.MutableSequence: list, typing.List: list, list: list, typing.Sequence: list, typing.Set: set, set: set, typing.MutableSet: set, collections.abc.MutableSet: set, collections.abc.Set: frozenset, typing.MutableMapping: dict, typing.Mapping: dict, collections.abc.Mapping: dict, collections.abc.MutableMapping: dict, float: float, int: int, str: str, bool: bool}
        values_type_origin = get_origin(values_source_type) or values_source_type
        instructions = 'set using `DefaultDict[..., Annotated[..., Field(default_factory=...)]]`'
        if isinstance(values_type_origin, TypeVar):

            def type_var_default_factory() -> None:
                raise RuntimeError('Generic defaultdict cannot be used without a concrete value type or an explicit default factory, ' + instructions)
            return type_var_default_factory
        elif values_type_origin not in allowed_default_types:
            allowed_msg = ', '.join([t.__name__ for t in set(allowed_default_types.values())])
            raise PydanticSchemaGenerationError(f'Unable to infer a default factory for keys of type {values_source_type}. Only {allowed_msg} are supported, other types require an explicit default factory ' + instructions)
        return allowed_default_types[values_type_origin]
    if _typing_extra.is_annotated(values_source_type):
        field_info = next((v for v in get_args(values_source_type) if isinstance(v, FieldInfo)), None)
    else:
        field_info = None
    if field_info and field_info.default_factory:
        default_default_factory = field_info.default_factory
    else:
        default_default_factory = infer_default()
    return default_default_factory