import numbers
from typing import Any, Union
import numpy as np
def assert_is_zero_or_one(x: Any) -> int:
    """Asserts that x is 0 or 1 and returns it as an int."""
    if not isinstance(x, numbers.Integral) or x < 0 or x > 1:
        raise TypeError('Not a boolean: %s' % x)
    return int(x)