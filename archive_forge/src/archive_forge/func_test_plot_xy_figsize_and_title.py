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
def test_plot_xy_figsize_and_title(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=5, freq='B'))
    ax = df.plot(x=1, y=2, title='Test', figsize=(16, 8))
    _check_text_labels(ax.title, 'Test')
    _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16.0, 8.0))