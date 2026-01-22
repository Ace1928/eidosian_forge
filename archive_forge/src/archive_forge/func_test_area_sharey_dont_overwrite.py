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
def test_area_sharey_dont_overwrite(self):
    df = DataFrame(np.random.default_rng(2).random((4, 2)), columns=['x', 'y'])
    fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)
    df.plot(ax=ax1, kind='area')
    df.plot(ax=ax2, kind='area')
    assert get_y_axis(ax1).joined(ax1, ax2)
    assert get_y_axis(ax2).joined(ax1, ax2)