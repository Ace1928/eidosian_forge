from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_short_data_repr_html(dataarray: xr.DataArray) -> None:
    data_repr = fh.short_data_repr_html(dataarray)
    assert data_repr.startswith('<pre>array')