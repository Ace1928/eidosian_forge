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
def dict_schema(self, schema: core_schema.DictSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a dict schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    json_schema: JsonSchemaValue = {'type': 'object'}
    keys_schema = self.generate_inner(schema['keys_schema']).copy() if 'keys_schema' in schema else {}
    keys_pattern = keys_schema.pop('pattern', None)
    values_schema = self.generate_inner(schema['values_schema']).copy() if 'values_schema' in schema else {}
    values_schema.pop('title', None)
    if values_schema or keys_pattern is not None:
        if keys_pattern is None:
            json_schema['additionalProperties'] = values_schema
        else:
            json_schema['patternProperties'] = {keys_pattern: values_schema}
    self.update_with_validations(json_schema, schema, self.ValidationsMapping.object)
    return json_schema