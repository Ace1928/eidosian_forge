import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_boxplot_legacy2(self):
    tuples = zip(string.ascii_letters[:10], range(10))
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=MultiIndex.from_tuples(tuples))
    grouped = df.groupby(level=1)
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        axes = _check_plot_works(grouped.boxplot, return_type='axes')
    _check_axes_shape(list(axes.values), axes_num=10, layout=(4, 3))