from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def handle_typed_dict_schema(self, schema: core_schema.TypedDictSchema, f: Walk) -> core_schema.CoreSchema:
    extras_schema = schema.get('extras_schema')
    if extras_schema is not None:
        schema['extras_schema'] = self.walk(extras_schema, f)
    replaced_computed_fields: list[core_schema.ComputedField] = []
    for computed_field in schema.get('computed_fields', ()):
        replaced_field = computed_field.copy()
        replaced_field['return_schema'] = self.walk(computed_field['return_schema'], f)
        replaced_computed_fields.append(replaced_field)
    if replaced_computed_fields:
        schema['computed_fields'] = replaced_computed_fields
    replaced_fields: dict[str, core_schema.TypedDictField] = {}
    for k, v in schema['fields'].items():
        replaced_field = v.copy()
        replaced_field['schema'] = self.walk(v['schema'], f)
        replaced_fields[k] = replaced_field
    schema['fields'] = replaced_fields
    return schema