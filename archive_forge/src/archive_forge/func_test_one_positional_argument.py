import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_one_positional_argument():
    with tm.assert_produces_warning(FutureWarning):
        assert h(23) == 23