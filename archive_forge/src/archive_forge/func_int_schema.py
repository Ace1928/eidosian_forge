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
def int_schema(self, schema: core_schema.IntSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches an int value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    json_schema: dict[str, Any] = {'type': 'integer'}
    self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
    json_schema = {k: v for k, v in json_schema.items() if v not in {math.inf, -math.inf}}
    return json_schema