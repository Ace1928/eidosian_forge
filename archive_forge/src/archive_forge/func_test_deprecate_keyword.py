import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@pytest.mark.parametrize('key', ['old', 'unchanged'])
def test_deprecate_keyword(key):
    x = 9
    if key == 'old':
        klass = FutureWarning
        expected = (x, True)
    else:
        klass = None
        expected = (True, x)
    with tm.assert_produces_warning(klass):
        assert _f4(**{key: x}) == expected