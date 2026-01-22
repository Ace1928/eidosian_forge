import collections.abc
import math
import warnings
from typing import cast, List, Optional, Tuple, Union
import torch
def modify_low_high(low: Optional[float], high: Optional[float], *, lowest_inclusive: float, highest_exclusive: float, default_low: float, default_high: float) -> Tuple[float, float]:
    """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high)
        if required.
        """

    def clamp(a: float, l: float, h: float) -> float:
        return min(max(a, l), h)
    low = low if low is not None else default_low
    high = high if high is not None else default_high
    if any((isinstance(value, float) and math.isnan(value) for value in [low, high])):
        raise ValueError(f'`low` and `high` cannot be NaN, but got low={low!r} and high={high!r}')
    elif low == high and dtype in _FLOATING_OR_COMPLEX_TYPES:
        warnings.warn('Passing `low==high` to `torch.testing.make_tensor` for floating or complex types is deprecated since 2.1 and will be removed in 2.3. Use torch.full(...) instead.', FutureWarning)
    elif low >= high:
        raise ValueError(f'`low` must be less than `high`, but got {low} >= {high}')
    elif high < lowest_inclusive or low >= highest_exclusive:
        raise ValueError(f'The value interval specified by `low` and `high` is [{low}, {high}), but {dtype} only supports [{lowest_inclusive}, {highest_exclusive})')
    low = clamp(low, lowest_inclusive, highest_exclusive)
    high = clamp(high, lowest_inclusive, highest_exclusive)
    if dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        return (math.ceil(low), math.ceil(high))
    return (low, high)