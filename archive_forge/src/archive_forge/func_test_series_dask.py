from packaging.version import Version
from unittest import SkipTest
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import pytest
from hvplot import hvPlotTabular
from hvplot.tests.util import makeDataFrame
@pytest.mark.skipif(dd is None, reason='dask not installed')
@series_kinds
def test_series_dask(kind):
    ser = dd.from_pandas(pd.Series(np.random.rand(10), name='A'), npartitions=2)
    assert isinstance(ser, dd.Series)
    ser.hvplot(kind=kind)