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
@pytest.mark.parametrize('indexer', [tm.getitem, tm.loc])
def test_index_type_coercion(self, indexer):
    for s in [Series(range(5)), Series(range(5), index=range(1, 6))]:
        assert is_integer_dtype(s.index)
        s2 = s.copy()
        indexer(s2)[0.1] = 0
        assert is_float_dtype(s2.index)
        assert indexer(s2)[0.1] == 0
        s2 = s.copy()
        indexer(s2)[0.0] = 0
        exp = s.index
        if 0 not in s:
            exp = Index(s.index.tolist() + [0])
        tm.assert_index_equal(s2.index, exp)
        s2 = s.copy()
        indexer(s2)['0'] = 0
        assert is_object_dtype(s2.index)
    for s in [Series(range(5), index=np.arange(5.0))]:
        assert is_float_dtype(s.index)
        s2 = s.copy()
        indexer(s2)[0.1] = 0
        assert is_float_dtype(s2.index)
        assert indexer(s2)[0.1] == 0
        s2 = s.copy()
        indexer(s2)[0.0] = 0
        tm.assert_index_equal(s2.index, s.index)
        s2 = s.copy()
        indexer(s2)['0'] = 0
        assert is_object_dtype(s2.index)