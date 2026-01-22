from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def inline_refs(s: core_schema.CoreSchema, recurse: Recurse) -> core_schema.CoreSchema:
    if s['type'] == 'definition-ref':
        ref = s['schema_ref']
        if can_be_inlined(s, ref):
            new = definitions.pop(ref)
            ref_counts[ref] -= 1
            if 'serialization' in s:
                new['serialization'] = s['serialization']
            s = recurse(new, inline_refs)
            return s
        else:
            return recurse(s, inline_refs)
    else:
        return recurse(s, inline_refs)