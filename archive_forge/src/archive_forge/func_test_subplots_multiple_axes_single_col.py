import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_subplots_multiple_axes_single_col(self):
    _, axes = mpl.pyplot.subplots(1, 1)
    df = DataFrame(np.random.default_rng(2).random((10, 1)), index=list(string.ascii_letters[:10]))
    axes = df.plot(subplots=True, ax=[axes], sharex=False, sharey=False)
    _check_axes_shape(axes, axes_num=1, layout=(1, 1))
    assert axes.shape == (1,)