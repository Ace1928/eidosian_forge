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
@pytest.mark.parametrize('x, y', [('x', 'y'), ('y', 'x'), ('y', 'y')])
def test_plot_scatter_with_categorical_data(self, x, y):
    df = DataFrame({'x': [1, 2, 3, 4], 'y': pd.Categorical(['a', 'b', 'a', 'c'])})
    _check_plot_works(df.plot.scatter, x=x, y=y)