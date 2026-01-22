from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
@property
def month_deltas(self):
    """Sorted unique month deltas."""
    if self._month_deltas is None:
        self._month_deltas = _unique_deltas(self.index.year * 12 + self.index.month)
    return self._month_deltas