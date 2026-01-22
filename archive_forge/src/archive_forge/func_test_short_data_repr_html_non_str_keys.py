from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
def test_short_data_repr_html_non_str_keys(dataset: xr.Dataset) -> None:
    ds = dataset.assign({2: lambda x: x['tmin']})
    fh.dataset_repr(ds)