import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning')
def test_to_frame_expanddim(self):

    class SubclassedSeries(Series):

        @property
        def _constructor_expanddim(self):
            return SubclassedFrame

    class SubclassedFrame(DataFrame):
        pass
    ser = SubclassedSeries([1, 2, 3], name='X')
    result = ser.to_frame()
    assert isinstance(result, SubclassedFrame)
    expected = SubclassedFrame({'X': [1, 2, 3]})
    tm.assert_frame_equal(result, expected)