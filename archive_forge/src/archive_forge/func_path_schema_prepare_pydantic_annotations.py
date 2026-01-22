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
def path_schema_prepare_pydantic_annotations(source_type: Any, annotations: Iterable[Any], _config: ConfigDict) -> tuple[Any, list[Any]] | None:
    import pathlib
    if source_type not in {os.PathLike, pathlib.Path, pathlib.PurePath, pathlib.PosixPath, pathlib.PurePosixPath, pathlib.PureWindowsPath}:
        return None
    metadata, remaining_annotations = _known_annotated_metadata.collect_known_metadata(annotations)
    _known_annotated_metadata.check_metadata(metadata, _known_annotated_metadata.STR_CONSTRAINTS, source_type)
    construct_path = pathlib.PurePath if source_type is os.PathLike else source_type

    def path_validator(input_value: str) -> os.PathLike[Any]:
        try:
            return construct_path(input_value)
        except TypeError as e:
            raise PydanticCustomError('path_type', 'Input is not a valid path') from e
    constrained_str_schema = core_schema.str_schema(**metadata)
    instance_schema = core_schema.json_or_python_schema(json_schema=core_schema.no_info_after_validator_function(path_validator, constrained_str_schema), python_schema=core_schema.is_instance_schema(source_type))
    strict: bool | None = None
    for annotation in annotations:
        if isinstance(annotation, Strict):
            strict = annotation.strict
    schema = core_schema.lax_or_strict_schema(lax_schema=core_schema.union_schema([instance_schema, core_schema.no_info_after_validator_function(path_validator, constrained_str_schema)], custom_error_type='path_type', custom_error_message='Input is not a valid path', strict=True), strict_schema=instance_schema, serialization=core_schema.to_string_ser_schema(), strict=strict)
    return (source_type, [InnerSchemaValidator(schema, js_core_schema=constrained_str_schema, js_schema_update={'format': 'path'}), *remaining_annotations])