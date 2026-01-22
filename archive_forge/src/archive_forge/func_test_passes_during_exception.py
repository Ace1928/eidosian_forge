import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_passes_during_exception():
    with pytest.raises(SyntaxError, match='Error'):
        with tm.assert_produces_warning(None):
            raise SyntaxError('Error')
    with pytest.raises(ValueError, match='Error'):
        with tm.assert_produces_warning(FutureWarning, match='FutureWarning'):
            warnings.warn('FutureWarning', FutureWarning)
            raise ValueError('Error')