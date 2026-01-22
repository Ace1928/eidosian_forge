import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('df1,df2', [(DataFrame({'A': [1, 2, 3]}), DataFrame({'A': [1, 2, 3, 4]})), (DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), DataFrame({'A': [1, 2, 3]}))])
def test_frame_equal_shape_mismatch(df1, df2, obj_fixture):
    msg = f'{obj_fixture} are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, obj=obj_fixture)