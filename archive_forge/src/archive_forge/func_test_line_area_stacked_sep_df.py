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
@pytest.mark.parametrize('kind', ['line', 'area'])
def test_line_area_stacked_sep_df(self, kind):
    sep_df = DataFrame({'w': np.random.default_rng(2).random(6), 'x': np.random.default_rng(2).random(6), 'y': -np.random.default_rng(2).random(6), 'z': -np.random.default_rng(2).random(6)})
    ax1 = _check_plot_works(sep_df.plot, kind=kind, stacked=False)
    ax2 = _check_plot_works(sep_df.plot, kind=kind, stacked=True)
    self._compare_stacked_y_cood(ax1.lines[:2], ax2.lines[:2])
    self._compare_stacked_y_cood(ax1.lines[2:], ax2.lines[2:])