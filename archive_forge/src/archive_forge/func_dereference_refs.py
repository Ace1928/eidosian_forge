from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set
def dereference_refs(schema_obj: dict, *, full_schema: Optional[dict]=None, skip_keys: Optional[Sequence[str]]=None) -> dict:
    """Try to substitute $refs in JSON Schema."""
    full_schema = full_schema or schema_obj
    skip_keys = skip_keys if skip_keys is not None else _infer_skip_keys(schema_obj, full_schema)
    return _dereference_refs_helper(schema_obj, full_schema, skip_keys)