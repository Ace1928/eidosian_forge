from __future__ import annotations
from enum import Enum
from typing import Literal
import pandas as pd
from packaging.version import Version
from xarray.coding import cftime_offsets
def count_not_none(*args) -> int:
    """Compute the number of non-None arguments.

    Copied from pandas.core.common.count_not_none (not part of the public API)
    """
    return sum((arg is not None for arg in args))