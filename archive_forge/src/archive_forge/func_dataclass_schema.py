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
def dataclass_schema(self, schema: core_schema.DataclassSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a dataclass.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    cls = schema['cls']
    config: ConfigDict = getattr(cls, '__pydantic_config__', cast('ConfigDict', {}))
    title = config.get('title') or cls.__name__
    with self._config_wrapper_stack.push(config):
        json_schema = self.generate_inner(schema['schema']).copy()
    json_schema_extra = config.get('json_schema_extra')
    json_schema = self._update_class_schema(json_schema, title, config.get('extra', None), cls, json_schema_extra)
    if is_dataclass(cls) and (not hasattr(cls, '__pydantic_validator__')):
        description = None
    else:
        description = None if cls.__doc__ is None else inspect.cleandoc(cls.__doc__)
    if description:
        json_schema['description'] = description
    return json_schema