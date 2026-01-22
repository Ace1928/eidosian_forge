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
def test_ensure_index_uint64(self):
    values = [0, np.iinfo(np.uint64).max]
    result = ensure_index(values)
    assert list(result) == values
    expected = Index(values, dtype='uint64')
    tm.assert_index_equal(result, expected)