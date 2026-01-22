import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_tight_layout(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((100, 2)))
    df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000, 819331200000000000, size=100, dtype=np.int64))
    _check_plot_works(df.hist, default_axes=True)
    mpl.pyplot.tight_layout()