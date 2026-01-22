import numpy as np
import pytest
from pandas import Index
import pandas._testing as tm
def test_insert_missing(self, nulls_fixture, using_infer_string):
    expected = Index(['a', nulls_fixture, 'b', 'c'], dtype=object)
    result = Index(list('abc'), dtype=object).insert(1, Index([nulls_fixture], dtype=object))
    tm.assert_index_equal(result, expected)