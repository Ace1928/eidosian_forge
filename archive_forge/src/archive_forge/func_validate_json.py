from __future__ import annotations as _annotations
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Set, TypeVar, Union, cast, final, overload
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import Literal, get_args, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel
from ._internal import _config, _generate_schema, _typing_extra
from .config import ConfigDict
from .json_schema import (
from .plugin._schema_validator import create_schema_validator
def validate_json(self, __data: str | bytes, *, strict: bool | None=None, context: dict[str, Any] | None=None) -> T:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/json/#json-parsing

        Validate a JSON string or bytes against the model.

        Args:
            __data: The JSON data to validate against the model.
            strict: Whether to strictly check types.
            context: Additional context to use during validation.

        Returns:
            The validated object.
        """
    return self.validator.validate_json(__data, strict=strict, context=context)