import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_3d_xarray_dataset_with_coords(self):
    import pandas as pd
    data, x, y, by, groupby = process_xarray(data=self.ds, **self.default_kwargs)
    assert isinstance(data, pd.DataFrame)
    assert x == 'time'
    assert y == ['air']
    assert not by
    assert groupby == ['lon', 'lat']