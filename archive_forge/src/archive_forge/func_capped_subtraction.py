import numbers
from typing import Any, Union
import numpy as np
def capped_subtraction(x: int, y: int) -> int:
    """Saturated arithmetics. Returns x - y truncated to the int64_t range."""
    assert_is_int64(x)
    assert_is_int64(y)
    if y == 0:
        return x
    if x == y:
        if x == INT_MAX or x == INT_MIN:
            raise OverflowError('Integer NaN: subtracting INT_MAX or INT_MIN to itself')
        return 0
    if x == INT_MAX or x == INT_MIN:
        return x
    if y == INT_MAX:
        return INT_MIN
    if y == INT_MIN:
        return INT_MAX
    return to_capped_int64(x - y)