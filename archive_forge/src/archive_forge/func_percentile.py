import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
def percentile(input_values, key=lambda x: x):
    if not input_values:
        return None
    input_values = sorted(input_values)
    k = (len(input_values) - 1) * self._q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(input_values[int(k)])
    d0 = key(input_values[int(f)]) * (c - k)
    d1 = key(input_values[int(c)]) * (k - f)
    return round(d0 + d1, 5)