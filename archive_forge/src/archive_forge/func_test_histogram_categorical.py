import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_histogram_categorical(self):
    series = Dataset(pd.Series(['A', 'B', 'C']))
    kwargs = {'bin_range': ('A', 'C'), 'normed': False, 'cumulative': False, 'num_bins': 3}
    with pytest.raises(ValueError):
        histogram(series, **kwargs)