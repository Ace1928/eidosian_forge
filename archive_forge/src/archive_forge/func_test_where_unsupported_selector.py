from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('selector', [ds.any(), ds.count(), ds.mean('value'), ds.std('value'), ds.sum('value'), ds.summary(any=ds.any()), ds.var('value'), ds.where(ds.max('value'), 'other')])
def test_where_unsupported_selector(selector):
    cvs = ds.Canvas(plot_width=10, plot_height=10)
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 2], value=[1, 2]))
    with pytest.raises(TypeError, match='selector can only be a first, first_n, last, last_n, max, max_n, min or min_n reduction'):
        cvs.line(df, 'x', 'y', agg=ds.where(selector, 'value'))