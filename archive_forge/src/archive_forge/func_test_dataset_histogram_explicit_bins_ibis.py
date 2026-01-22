import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
@ibis_skip
@pytest.mark.usefixtures('ibis_sqlite_backend')
def test_dataset_histogram_explicit_bins_ibis(self):
    df = pd.DataFrame(dict(x=np.arange(10)))
    t = ibis.memtable(df, name='t')
    ds = Dataset(t, vdims='x')
    op_hist = histogram(ds, bins=[0, 1, 3], normed=False)
    hist = Histogram(([0, 1, 3], [1, 3]), vdims=('x_count', 'Count'))
    self.assertEqual(op_hist, hist)