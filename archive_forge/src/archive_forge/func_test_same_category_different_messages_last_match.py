import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_same_category_different_messages_last_match():
    category = DeprecationWarning
    with tm.assert_produces_warning(category, match='^Match this'):
        warnings.warn('Do not match that', category)
        warnings.warn('Do not match that either', category)
        warnings.warn('Match this', category)