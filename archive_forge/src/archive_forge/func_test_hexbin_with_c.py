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
@pytest.mark.parametrize('reduce_C', [None, np.std])
def test_hexbin_with_c(self, reduce_C):
    df = DataFrame({'A': np.random.default_rng(2).uniform(size=20), 'B': np.random.default_rng(2).uniform(size=20), 'C': np.arange(20) + np.random.default_rng(2).uniform(size=20)})
    ax = df.plot.hexbin(x='A', y='B', C='C', reduce_C_function=reduce_C)
    assert len(ax.collections) == 1