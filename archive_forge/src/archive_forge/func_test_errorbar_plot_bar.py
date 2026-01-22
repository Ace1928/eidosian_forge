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
@pytest.mark.slow
def test_errorbar_plot_bar(self):
    d = {'x': np.arange(12), 'y': np.arange(12, 0, -1)}
    df = DataFrame(d)
    d_err = {'x': np.ones(12) * 0.2, 'y': np.ones(12) * 0.4}
    df_err = DataFrame(d_err)
    ax = _check_plot_works((df + 1).plot, yerr=df_err, xerr=df_err, kind='bar', log=True)
    _check_has_errorbars(ax, xerr=2, yerr=2)