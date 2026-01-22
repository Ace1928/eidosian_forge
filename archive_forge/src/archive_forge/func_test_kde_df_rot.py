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
def test_kde_df_rot(self):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    ax = df.plot(kind='kde', rot=20, fontsize=5)
    _check_ticks_props(ax, xrot=20, xlabelsize=5, ylabelsize=5)