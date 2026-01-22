from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
@property
def deltas(self):
    """Sorted unique timedeltas as microseconds."""
    if self._deltas is None:
        self._deltas = _unique_deltas(self.values)
    return self._deltas