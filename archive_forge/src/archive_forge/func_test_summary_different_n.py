from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('df', dfs)
def test_summary_different_n(df):
    msg = 'Using multiple FloatingNReductions with different n values is not supported'
    with pytest.raises(ValueError, match=msg):
        c.points(df, 'x', 'y', ds.summary(min_n=ds.where(ds.min_n('plusminus', 2)), max_n=ds.where(ds.max_n('plusminus', 3))))