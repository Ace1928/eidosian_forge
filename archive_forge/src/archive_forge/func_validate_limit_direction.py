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
def validate_limit_direction(limit_direction: str) -> Literal['forward', 'backward', 'both']:
    valid_limit_directions = ['forward', 'backward', 'both']
    limit_direction = limit_direction.lower()
    if limit_direction not in valid_limit_directions:
        raise ValueError(f"Invalid limit_direction: expecting one of {valid_limit_directions}, got '{limit_direction}'.")
    return limit_direction