from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def handle_call_schema(self, schema: core_schema.CallSchema, f: Walk) -> core_schema.CoreSchema:
    schema['arguments_schema'] = self.walk(schema['arguments_schema'], f)
    if 'return_schema' in schema:
        schema['return_schema'] = self.walk(schema['return_schema'], f)
    return schema