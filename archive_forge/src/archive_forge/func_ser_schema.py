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
def ser_schema(self, schema: core_schema.SerSchema | core_schema.IncExSeqSerSchema | core_schema.IncExDictSerSchema) -> JsonSchemaValue | None:
    """Generates a JSON schema that matches a schema that defines a serialized object.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    schema_type = schema['type']
    if schema_type == 'function-plain' or schema_type == 'function-wrap':
        return_schema = schema.get('return_schema')
        if return_schema is not None:
            return self.generate_inner(return_schema)
    elif schema_type == 'format' or schema_type == 'to-string':
        return self.str_schema(core_schema.str_schema())
    elif schema['type'] == 'model':
        return self.generate_inner(schema['schema'])
    return None