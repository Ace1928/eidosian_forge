import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
@pytest.fixture(params=[(RuntimeWarning, UserWarning), (UserWarning, FutureWarning), (FutureWarning, RuntimeWarning), (DeprecationWarning, PerformanceWarning), (PerformanceWarning, FutureWarning), (DtypeWarning, DeprecationWarning), (ResourceWarning, DeprecationWarning), (FutureWarning, DeprecationWarning)], ids=lambda x: type(x).__name__)
def pair_different_warnings(request):
    """
    Return pair or different warnings.

    Useful for testing how several different warnings are handled
    in tm.assert_produces_warning.
    """
    return request.param