import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_out_of_scope(using_copy_on_write):

    def func():
        df = DataFrame({'a': [1, 2], 'b': 1.5, 'c': 1})
        result = df[['a', 'b']]
        return result
    result = func()
    if using_copy_on_write:
        assert not result._mgr.blocks[0].refs.has_reference()
        assert not result._mgr.blocks[1].refs.has_reference()