from __future__ import annotations
from typing import Any, Dict, List, Optional
def merge_lists(left: Optional[List], right: Optional[List]) -> Optional[List]:
    """Add two lists, handling None."""
    if left is None and right is None:
        return None
    elif left is None or right is None:
        return left or right
    else:
        merged = left.copy()
        for e in right:
            if isinstance(e, dict) and 'index' in e and isinstance(e['index'], int):
                to_merge = [i for i, e_left in enumerate(merged) if e_left['index'] == e['index']]
                if to_merge:
                    merged[to_merge[0]] = merge_dicts(merged[to_merge[0]], e)
                else:
                    merged = merged + [e]
            else:
                merged = merged + [e]
        return merged