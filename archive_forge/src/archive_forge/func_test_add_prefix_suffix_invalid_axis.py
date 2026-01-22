import pytest
from pandas import Index
import pandas._testing as tm
def test_add_prefix_suffix_invalid_axis(float_frame):
    with pytest.raises(ValueError, match='No axis named 2 for object type DataFrame'):
        float_frame.add_prefix('foo#', axis=2)
    with pytest.raises(ValueError, match='No axis named 2 for object type DataFrame'):
        float_frame.add_suffix('foo#', axis=2)