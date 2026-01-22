from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def handle_dict_schema(self, schema: core_schema.DictSchema, f: Walk) -> core_schema.CoreSchema:
    keys_schema = schema.get('keys_schema')
    if keys_schema is not None:
        schema['keys_schema'] = self.walk(keys_schema, f)
    values_schema = schema.get('values_schema')
    if values_schema:
        schema['values_schema'] = self.walk(values_schema, f)
    return schema