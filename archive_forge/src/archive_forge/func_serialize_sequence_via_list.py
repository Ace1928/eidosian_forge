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
def serialize_sequence_via_list(self, v: Any, handler: core_schema.SerializerFunctionWrapHandler, info: core_schema.SerializationInfo) -> Any:
    items: list[Any] = []
    for index, item in enumerate(v):
        try:
            v = handler(item, index)
        except PydanticOmit:
            pass
        else:
            items.append(v)
    if info.mode_is_json():
        return items
    else:
        return self.mapped_origin(items)