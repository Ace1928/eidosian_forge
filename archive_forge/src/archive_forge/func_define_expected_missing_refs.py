from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def define_expected_missing_refs(schema: core_schema.CoreSchema, allowed_missing_refs: set[str]) -> core_schema.CoreSchema | None:
    if not allowed_missing_refs:
        return None
    refs = collect_definitions(schema).keys()
    expected_missing_refs = allowed_missing_refs.difference(refs)
    if expected_missing_refs:
        definitions: list[core_schema.CoreSchema] = [core_schema.none_schema(ref=ref, metadata={HAS_INVALID_SCHEMAS_METADATA_KEY: True}) for ref in expected_missing_refs]
        return core_schema.definitions_schema(schema, definitions)
    return None