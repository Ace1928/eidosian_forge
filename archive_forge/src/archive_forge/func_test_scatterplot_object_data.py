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
@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('x, y', [('a', 'b'), (0, 1)])
@pytest.mark.parametrize('b_col', [[2, 3, 4], ['a', 'b', 'c']])
def test_scatterplot_object_data(self, b_col, x, y, infer_string):
    with option_context('future.infer_string', infer_string):
        df = DataFrame({'a': ['A', 'B', 'C'], 'b': b_col})
        _check_plot_works(df.plot.scatter, x=x, y=y)