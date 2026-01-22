from __future__ import annotations
import datetime
from typing import TypedDict
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core.pdcompat import _convert_base_to_offset
from xarray.core.resample_cftime import CFTimeGrouper
@pytest.mark.parametrize(('freq', 'expected'), [('s', 'left'), ('min', 'left'), ('h', 'left'), ('D', 'left'), ('ME', 'right'), ('MS', 'left'), ('QE', 'right'), ('QS', 'left'), ('YE', 'right'), ('YS', 'left')])
def test_closed_label_defaults(freq, expected) -> None:
    assert CFTimeGrouper(freq=freq).closed == expected
    assert CFTimeGrouper(freq=freq).label == expected