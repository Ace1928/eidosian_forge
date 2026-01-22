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
def test_unique_na(self):
    idx = Index([2, np.nan, 2, 1], name='my_index')
    expected = Index([2, np.nan, 1], name='my_index')
    result = idx.unique()
    tm.assert_index_equal(result, expected)