from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_rotation_default(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    _, ax = mpl.pyplot.subplots()
    axes = df.plot(ax=ax)
    _check_ticks_props(axes, xrot=0)