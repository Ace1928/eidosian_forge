from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@td.skip_if_no('pyarrow')
def test_isin_arrow_string_null(self):
    index = Index(['a', 'b'], dtype='string[pyarrow_numpy]')
    result = index.isin([None])
    expected = np.array([False, False])
    tm.assert_numpy_array_equal(result, expected)