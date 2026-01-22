import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
@pytest.mark.parametrize('val', [1, 2.0, None])
def test_cast_1d_array_invalid_scalar(val):
    with pytest.raises(TypeError, match='has no len()'):
        construct_1d_object_array_from_listlike(val)