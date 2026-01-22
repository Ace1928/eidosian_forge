from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def count_refs(s: core_schema.CoreSchema, recurse: Recurse) -> core_schema.CoreSchema:
    if s['type'] != 'definition-ref':
        return recurse(s, count_refs)
    ref = s['schema_ref']
    ref_counts[ref] += 1
    if ref_counts[ref] >= 2:
        if current_recursion_ref_count[ref] != 0:
            involved_in_recursion[ref] = True
        return s
    current_recursion_ref_count[ref] += 1
    recurse(definitions[ref], count_refs)
    current_recursion_ref_count[ref] -= 1
    return s