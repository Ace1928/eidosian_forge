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
def typed_dict_schema(self, schema: core_schema.TypedDictSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a typed dict.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    total = schema.get('total', True)
    named_required_fields: list[tuple[str, bool, CoreSchemaField]] = [(name, self.field_is_required(field, total), field) for name, field in schema['fields'].items() if self.field_is_present(field)]
    if self.mode == 'serialization':
        named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
    config = _get_typed_dict_config(schema)
    with self._config_wrapper_stack.push(config):
        json_schema = self._named_required_fields_schema(named_required_fields)
    extra = schema.get('extra_behavior')
    if extra is None:
        extra = config.get('extra', 'ignore')
    if extra == 'forbid':
        json_schema['additionalProperties'] = False
    elif extra == 'allow':
        json_schema['additionalProperties'] = True
    return json_schema