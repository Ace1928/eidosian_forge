import numbers
from typing import Any, Union
import numpy as np
def to_capped_int64(v: int) -> int:
    """Restrict v within [INT_MIN..INT_MAX] range."""
    if v > INT_MAX:
        return INT_MAX
    if v < INT_MIN:
        return INT_MIN
    return v