import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
@pytest.mark.parametrize('ser, keys', [(Series([10]), (0, 0)), (Series([1, 2, 3], index=list('abc')), (0, 1))])
def test_ser_tup_indexer_exceeds_dimensions(ser, keys, indexer_li):
    exp_err, exp_msg = (IndexingError, 'Too many indexers')
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys]
    if indexer_li == tm.iloc:
        exp_err, exp_msg = (IndexError, 'too many indices for array')
    with pytest.raises(exp_err, match=exp_msg):
        indexer_li(ser)[keys] = 0