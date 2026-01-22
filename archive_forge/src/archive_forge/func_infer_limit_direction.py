from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def infer_limit_direction(limit_direction: Literal['backward', 'forward', 'both'] | None, method: str) -> Literal['backward', 'forward', 'both']:
    if limit_direction is None:
        if method in ('backfill', 'bfill'):
            limit_direction = 'backward'
        else:
            limit_direction = 'forward'
    else:
        if method in ('pad', 'ffill') and limit_direction != 'forward':
            raise ValueError(f"`limit_direction` must be 'forward' for method `{method}`")
        if method in ('backfill', 'bfill') and limit_direction != 'backward':
            raise ValueError(f"`limit_direction` must be 'backward' for method `{method}`")
    return limit_direction