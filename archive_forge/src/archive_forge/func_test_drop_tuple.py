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
@pytest.mark.parametrize('values', [['a', 'b', ('c', 'd')], ['a', ('c', 'd'), 'b'], [('c', 'd'), 'a', 'b']])
@pytest.mark.parametrize('to_drop', [[('c', 'd'), 'a'], ['a', ('c', 'd')]])
def test_drop_tuple(self, values, to_drop):
    index = Index(values)
    expected = Index(['b'], dtype=object)
    result = index.drop(to_drop)
    tm.assert_index_equal(result, expected)
    removed = index.drop(to_drop[0])
    for drop_me in (to_drop[1], [to_drop[1]]):
        result = removed.drop(drop_me)
        tm.assert_index_equal(result, expected)
    removed = index.drop(to_drop[1])
    msg = f'\\"\\[{re.escape(to_drop[1].__repr__())}\\] not found in axis\\"'
    for drop_me in (to_drop[1], [to_drop[1]]):
        with pytest.raises(KeyError, match=msg):
            removed.drop(drop_me)