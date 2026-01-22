import copy
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, TypeVar, Union
from ray.util.annotations import Deprecated
@Deprecated
def merge_dicts(d1: dict, d2: dict) -> dict:
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged