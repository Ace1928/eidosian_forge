from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_short_data_repr_html_dask(dask_dataarray: xr.DataArray) -> None:
    assert hasattr(dask_dataarray.data, '_repr_html_')
    data_repr = fh.short_data_repr_html(dask_dataarray)
    assert data_repr == dask_dataarray.data._repr_html_()