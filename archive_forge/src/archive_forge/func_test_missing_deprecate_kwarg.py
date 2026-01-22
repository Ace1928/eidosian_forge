import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@pytest.mark.parametrize('key', ['bogus', 12345, -1.23])
def test_missing_deprecate_kwarg(key):
    with tm.assert_produces_warning(FutureWarning):
        assert _f2(old=key) == key