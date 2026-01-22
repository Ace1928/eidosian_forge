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
def get_schema_from_definitions(self, json_ref: JsonRef) -> JsonSchemaValue | None:
    def_ref = self.json_to_defs_refs[json_ref]
    if def_ref in self._core_defs_invalid_for_json_schema:
        raise self._core_defs_invalid_for_json_schema[def_ref]
    return self.definitions.get(def_ref, None)