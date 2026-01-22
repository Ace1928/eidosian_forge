import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@pytest.mark.parametrize('key,klass', [('old', FutureWarning), ('new', None)])
def test_deprecate_kwarg(key, klass):
    x = 78
    with tm.assert_produces_warning(klass):
        assert _f1(**{key: x}) == x