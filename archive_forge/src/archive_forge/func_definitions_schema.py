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
def definitions_schema(self, schema: core_schema.DefinitionsSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a JSON object with definitions.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    for definition in schema['definitions']:
        try:
            self.generate_inner(definition)
        except PydanticInvalidForJsonSchema as e:
            core_ref: CoreRef = CoreRef(definition['ref'])
            self._core_defs_invalid_for_json_schema[self.get_defs_ref((core_ref, self.mode))] = e
            continue
    return self.generate_inner(schema['schema'])