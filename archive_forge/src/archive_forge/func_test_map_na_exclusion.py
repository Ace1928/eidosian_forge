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
def test_map_na_exclusion(self):
    index = Index([1.5, np.nan, 3, np.nan, 5])
    result = index.map(lambda x: x * 2, na_action='ignore')
    expected = index * 2
    tm.assert_index_equal(result, expected)