import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tuple_list', [[()], [(), ()]])
def test_construct_empty_tuples(self, tuple_list):
    result = Index(tuple_list)
    expected = MultiIndex.from_tuples(tuple_list)
    tm.assert_index_equal(result, expected)