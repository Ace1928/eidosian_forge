from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Unpack
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
@staticmethod
def merge_field_infos(*field_infos: FieldInfo, **overrides: Any) -> FieldInfo:
    """Merge `FieldInfo` instances keeping only explicitly set attributes.

        Later `FieldInfo` instances override earlier ones.

        Returns:
            FieldInfo: A merged FieldInfo instance.
        """
    flattened_field_infos: list[FieldInfo] = []
    for field_info in field_infos:
        flattened_field_infos.extend((x for x in field_info.metadata if isinstance(x, FieldInfo)))
        flattened_field_infos.append(field_info)
    field_infos = tuple(flattened_field_infos)
    if len(field_infos) == 1:
        field_info = copy(field_infos[0])
        field_info._attributes_set.update(overrides)
        for k, v in overrides.items():
            setattr(field_info, k, v)
        return field_info
    new_kwargs: dict[str, Any] = {}
    metadata = {}
    for field_info in field_infos:
        new_kwargs.update(field_info._attributes_set)
        for x in field_info.metadata:
            if not isinstance(x, FieldInfo):
                metadata[type(x)] = x
    new_kwargs.update(overrides)
    field_info = FieldInfo(**new_kwargs)
    field_info.metadata = list(metadata.values())
    return field_info