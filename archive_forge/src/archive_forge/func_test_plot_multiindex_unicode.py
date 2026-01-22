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
def test_plot_multiindex_unicode(self):
    index = MultiIndex.from_tuples([('α', 0), ('α', 1), ('β', 2), ('β', 3), ('γ', 4), ('γ', 5), ('δ', 6), ('δ', 7)], names=['i0', 'i1'])
    columns = MultiIndex.from_tuples([('bar', 'Δ'), ('bar', 'Ε')], names=['c0', 'c1'])
    df = DataFrame(np.random.default_rng(2).integers(0, 10, (8, 2)), columns=columns, index=index)
    _check_plot_works(df.plot, title='Σ')