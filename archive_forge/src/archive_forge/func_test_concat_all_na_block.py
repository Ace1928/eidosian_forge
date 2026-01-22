import itertools
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
@pytest.mark.parametrize('in_frame', [True, False])
def test_concat_all_na_block(self, data_missing, in_frame):
    valid_block = pd.Series(data_missing.take([1, 1]), index=[0, 1])
    na_block = pd.Series(data_missing.take([0, 0]), index=[2, 3])
    if in_frame:
        valid_block = pd.DataFrame({'a': valid_block})
        na_block = pd.DataFrame({'a': na_block})
    result = pd.concat([valid_block, na_block])
    if in_frame:
        expected = pd.DataFrame({'a': data_missing.take([1, 1, 0, 0])})
        tm.assert_frame_equal(result, expected)
    else:
        expected = pd.Series(data_missing.take([1, 1, 0, 0]))
        tm.assert_series_equal(result, expected)