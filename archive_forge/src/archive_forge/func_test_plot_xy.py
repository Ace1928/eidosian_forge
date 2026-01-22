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
def test_plot_xy(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=5, freq='B'))
    _check_data(df.plot(x=0, y=1), df.set_index('A')['B'].plot())
    _check_data(df.plot(x=0), df.set_index('A').plot())
    _check_data(df.plot(y=0), df.B.plot())
    _check_data(df.plot(x='A', y='B'), df.set_index('A').B.plot())
    _check_data(df.plot(x='A'), df.set_index('A').plot())
    _check_data(df.plot(y='B'), df.B.plot())