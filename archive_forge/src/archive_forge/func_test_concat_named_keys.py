import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_named_keys(self):
    df = DataFrame({'foo': [1, 2], 'bar': [0.1, 0.2]})
    index = Index(['a', 'b'], name='baz')
    concatted_named_from_keys = concat([df, df], keys=index)
    expected_named = DataFrame({'foo': [1, 2, 1, 2], 'bar': [0.1, 0.2, 0.1, 0.2]}, index=pd.MultiIndex.from_product((['a', 'b'], [0, 1]), names=['baz', None]))
    tm.assert_frame_equal(concatted_named_from_keys, expected_named)
    index_no_name = Index(['a', 'b'], name=None)
    concatted_named_from_names = concat([df, df], keys=index_no_name, names=['baz'])
    tm.assert_frame_equal(concatted_named_from_names, expected_named)
    concatted_unnamed = concat([df, df], keys=index_no_name)
    expected_unnamed = DataFrame({'foo': [1, 2, 1, 2], 'bar': [0.1, 0.2, 0.1, 0.2]}, index=pd.MultiIndex.from_product((['a', 'b'], [0, 1]), names=[None, None]))
    tm.assert_frame_equal(concatted_unnamed, expected_unnamed)