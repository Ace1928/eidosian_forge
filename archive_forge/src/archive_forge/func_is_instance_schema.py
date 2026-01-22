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
def is_instance_schema(self, schema: core_schema.IsInstanceSchema) -> JsonSchemaValue:
    """Handles JSON schema generation for a core schema that checks if a value is an instance of a class.

        Unless overridden in a subclass, this raises an error.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    return self.handle_invalid_for_json_schema(schema, f'core_schema.IsInstanceSchema ({schema['cls']})')