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
def test_subplots_multiple_axes(self):
    fig, axes = mpl.pyplot.subplots(2, 3)
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
    returned = df.plot(subplots=True, ax=axes[0], sharex=False, sharey=False)
    _check_axes_shape(returned, axes_num=3, layout=(1, 3))
    assert returned.shape == (3,)
    assert returned[0].figure is fig
    returned = df.plot(subplots=True, ax=axes[1], sharex=False, sharey=False)
    _check_axes_shape(returned, axes_num=3, layout=(1, 3))
    assert returned.shape == (3,)
    assert returned[0].figure is fig
    _check_axes_shape(axes, axes_num=6, layout=(2, 3))