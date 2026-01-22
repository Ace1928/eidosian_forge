from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
def build_schema_type_to_method(self) -> dict[CoreSchemaOrFieldType, Callable[[CoreSchemaOrField], JsonSchemaValue]]:
    """Builds a dictionary mapping fields to methods for generating JSON schemas.

        Returns:
            A dictionary containing the mapping of `CoreSchemaOrFieldType` to a handler method.

        Raises:
            TypeError: If no method has been defined for generating a JSON schema for a given pydantic core schema type.
        """
    mapping: dict[CoreSchemaOrFieldType, Callable[[CoreSchemaOrField], JsonSchemaValue]] = {}
    core_schema_types: list[CoreSchemaOrFieldType] = _typing_extra.all_literal_values(CoreSchemaOrFieldType)
    for key in core_schema_types:
        method_name = f'{key.replace('-', '_')}_schema'
        try:
            mapping[key] = getattr(self, method_name)
        except AttributeError as e:
            raise TypeError(f'No method for generating JsonSchema for core_schema.type={key!r} (expected: {type(self).__name__}.{method_name})') from e
    return mapping