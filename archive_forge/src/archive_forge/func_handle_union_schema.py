from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def handle_union_schema(self, schema: core_schema.UnionSchema, f: Walk) -> core_schema.CoreSchema:
    new_choices: list[CoreSchema | tuple[CoreSchema, str]] = []
    for v in schema['choices']:
        if isinstance(v, tuple):
            new_choices.append((self.walk(v[0], f), v[1]))
        else:
            new_choices.append(self.walk(v, f))
    schema['choices'] = new_choices
    return schema