import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_map_identity_mapping(index, request):
    result = index.map(lambda x: x)
    if index.dtype == object and result.dtype == bool:
        assert (index == result).all()
        return
    tm.assert_index_equal(result, index, exact='equiv')