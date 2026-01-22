from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_plot_scatter_shape(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['x', 'y', 'z', 'four'])
    axes = df.plot(x='x', y='y', kind='scatter', subplots=True)
    _check_axes_shape(axes, axes_num=1, layout=(1, 1))