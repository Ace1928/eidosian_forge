import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('errors,checker', [('raise', 'Unable to parse string "fail" at position 0'), ('ignore', lambda x: x == 'fail'), ('coerce', lambda x: np.isnan(x))])
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_scalar_fail(errors, checker):
    scalar = 'fail'
    if isinstance(checker, str):
        with pytest.raises(ValueError, match=checker):
            to_numeric(scalar, errors=errors)
    else:
        assert checker(to_numeric(scalar, errors=errors))