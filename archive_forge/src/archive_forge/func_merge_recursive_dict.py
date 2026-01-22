from __future__ import annotations
import collections.abc
from typing import Dict, Optional
def merge_recursive_dict(d: Dict, current_key: str) -> Dict:
    """
    Merge a recursive dictionary
    """
    if not isinstance(d, collections.abc.Mapping):
        return d
    mapping = {}
    for k, v in d.items():
        current_key = f'{current_key}.{k}'
        if isinstance(v, (collections.abc.Mapping, dict)):
            mapping[current_key] = merge_recursive_dict(v, current_key)
        else:
            mapping[current_key] = v
    return mapping