from collections import OrderedDict
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data_dict, orient, expected', [({}, 'index', RangeIndex(0)), ([{('a',): 1}, {('a',): 2}], 'columns', Index([('a',)], tupleize_cols=False)), ([OrderedDict([(('a',), 1), (('b',), 2)])], 'columns', Index([('a',), ('b',)], tupleize_cols=False)), ([{('a', 'b'): 1}], 'columns', Index([('a', 'b')], tupleize_cols=False))])
def test_constructor_from_dict_tuples(self, data_dict, orient, expected):
    df = DataFrame.from_dict(data_dict, orient)
    result = df.columns
    tm.assert_index_equal(result, expected)