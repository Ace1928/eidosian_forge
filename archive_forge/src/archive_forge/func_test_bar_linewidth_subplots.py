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
def test_bar_linewidth_subplots(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    axes = df.plot.bar(linewidth=2, subplots=True)
    _check_axes_shape(axes, axes_num=5, layout=(5, 1))
    for ax in axes:
        for r in ax.patches:
            assert r.get_linewidth() == 2