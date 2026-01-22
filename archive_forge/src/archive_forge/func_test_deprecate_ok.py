from textwrap import dedent
import pytest
from pandas.util._decorators import deprecate
import pandas._testing as tm
def test_deprecate_ok():
    depr_func = deprecate('depr_func', new_func, '1.0', msg='Use new_func instead.')
    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()
    assert result == 'new_func called'
    assert depr_func.__doc__ == dedent(new_func_with_deprecation.__doc__)