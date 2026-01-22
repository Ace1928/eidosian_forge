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
@pytest.mark.parametrize('index', ['string', 'datetime', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
def test_join_self(self, index, join_type):
    result = index.join(index, how=join_type)
    expected = index
    if join_type == 'outer':
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)