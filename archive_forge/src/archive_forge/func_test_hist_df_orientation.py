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
def test_hist_df_orientation(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    axes = df.plot.hist(rot=50, fontsize=8, orientation='horizontal')
    _check_ticks_props(axes, xrot=0, yrot=50, ylabelsize=8)