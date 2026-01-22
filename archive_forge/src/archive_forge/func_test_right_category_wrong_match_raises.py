import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_right_category_wrong_match_raises(pair_different_warnings):
    target_category, other_category = pair_different_warnings
    with pytest.raises(AssertionError, match='Did not see warning.*matching'):
        with tm.assert_produces_warning(target_category, match='^Match this'):
            warnings.warn('Do not match it', target_category)
            warnings.warn('Match this', other_category)