from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_getitem_index_namedtuple(self):
    IndexType = namedtuple('IndexType', ['a', 'b'])
    idx1 = IndexType('foo', 'bar')
    idx2 = IndexType('baz', 'bof')
    index = Index([idx1, idx2], name='composite_index', tupleize_cols=False)
    df = DataFrame([(1, 2), (3, 4)], index=index, columns=['A', 'B'])
    result = df.loc[IndexType('foo', 'bar')]['A']
    assert result == 1