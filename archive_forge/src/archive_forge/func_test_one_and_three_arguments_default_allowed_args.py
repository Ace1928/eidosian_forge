import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_one_and_three_arguments_default_allowed_args():
    with tm.assert_produces_warning(None):
        assert g(1, b=3, c=3, d=5) == 12