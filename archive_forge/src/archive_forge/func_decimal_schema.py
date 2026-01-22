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
def decimal_schema(self, schema: core_schema.DecimalSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a decimal value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    json_schema = self.str_schema(core_schema.str_schema())
    if self.mode == 'validation':
        multiple_of = schema.get('multiple_of')
        le = schema.get('le')
        ge = schema.get('ge')
        lt = schema.get('lt')
        gt = schema.get('gt')
        json_schema = {'anyOf': [self.float_schema(core_schema.float_schema(allow_inf_nan=schema.get('allow_inf_nan'), multiple_of=None if multiple_of is None else float(multiple_of), le=None if le is None else float(le), ge=None if ge is None else float(ge), lt=None if lt is None else float(lt), gt=None if gt is None else float(gt))), json_schema]}
    return json_schema