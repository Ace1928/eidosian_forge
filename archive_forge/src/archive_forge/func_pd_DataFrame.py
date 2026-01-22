from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def pd_DataFrame(*args, **kwargs):
    if kwargs.pop('geo', False):
        return sp.GeoDataFrame(*args, **kwargs)
    else:
        return pd.DataFrame(*args, **kwargs)