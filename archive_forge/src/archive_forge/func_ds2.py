import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.fixture(scope='module')
def ds2(da, da2):
    return xr.Dataset(dict(foo=da, bar=da2))