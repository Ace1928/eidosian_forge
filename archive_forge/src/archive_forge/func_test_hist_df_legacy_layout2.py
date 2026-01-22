import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_hist_df_legacy_layout2(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
    _check_plot_works(df.hist)