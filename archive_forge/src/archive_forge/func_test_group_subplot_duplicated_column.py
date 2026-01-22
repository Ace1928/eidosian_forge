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
def test_group_subplot_duplicated_column(self):
    d = {'a': np.arange(10), 'b': np.arange(10), 'c': np.arange(10)}
    df = DataFrame(d)
    with pytest.raises(ValueError, match='should be in only one subplot'):
        df.plot(subplots=[('a', 'b'), ('a', 'c')])