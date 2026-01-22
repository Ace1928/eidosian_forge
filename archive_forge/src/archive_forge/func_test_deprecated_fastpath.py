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
def test_deprecated_fastpath():
    msg = '[Uu]nexpected keyword argument'
    with pytest.raises(TypeError, match=msg):
        Index(np.array(['a', 'b'], dtype=object), name='test', fastpath=True)
    with pytest.raises(TypeError, match=msg):
        Index(np.array([1, 2, 3], dtype='int64'), name='test', fastpath=True)
    with pytest.raises(TypeError, match=msg):
        RangeIndex(0, 5, 2, name='test', fastpath=True)
    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(['a', 'b', 'c'], name='test', fastpath=True)