from __future__ import annotations
from typing import Any, Mapping
def remove_none_values(input_dict: Mapping[Any, Any]) -> dict[Any, Any]:
    """Remove all keys with None values from a dict."""
    new_dict = {}
    for key, val in input_dict.items():
        if isinstance(val, dict):
            val = remove_none_values(val)
        if val is not None:
            new_dict[key] = val
    return new_dict