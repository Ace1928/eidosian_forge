from __future__ import annotations as _annotations
import dataclasses
import typing
import warnings
from functools import partial, wraps
from typing import Any, Callable, ClassVar
from pydantic_core import (
from typing_extensions import TypeGuard
from ..errors import PydanticUndefinedAnnotation
from ..fields import FieldInfo
from ..plugin._schema_validator import create_schema_validator
from ..warnings import PydanticDeprecatedSince20
from . import _config, _decorators, _typing_extra
from ._fields import collect_dataclass_fields
from ._generate_schema import GenerateSchema
from ._generics import get_standard_typevars_map
from ._mock_val_ser import set_dataclass_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
def set_dataclass_fields(cls: type[StandardDataclass], types_namespace: dict[str, Any] | None=None) -> None:
    """Collect and set `cls.__pydantic_fields__`.

    Args:
        cls: The class.
        types_namespace: The types namespace, defaults to `None`.
    """
    typevars_map = get_standard_typevars_map(cls)
    fields = collect_dataclass_fields(cls, types_namespace, typevars_map=typevars_map)
    cls.__pydantic_fields__ = fields