import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@pytest.mark.parametrize('key', list(_f2_mappings.keys()))
def test_dict_deprecate_kwarg(key):
    with tm.assert_produces_warning(FutureWarning):
        assert _f2(old=key) == _f2_mappings[key]