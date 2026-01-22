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
def test_x_multiindex_values_ticks(self):
    index = MultiIndex.from_product([[2012, 2013], [1, 2]])
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['A', 'B'], index=index)
    ax = df.plot()
    ax.set_xlim(-1, 4)
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    labels_position = dict(zip(xticklabels, ax.get_xticks()))
    assert labels_position['(2012, 1)'] == 0.0
    assert labels_position['(2012, 2)'] == 1.0
    assert labels_position['(2013, 1)'] == 2.0
    assert labels_position['(2013, 2)'] == 3.0