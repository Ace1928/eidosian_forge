from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('cast_index', [True, False])
@pytest.mark.parametrize('vals', [[True, False, True], np.array([True, False, True], dtype=bool)])
def test_constructor_dtypes_to_object(self, cast_index, vals):
    if cast_index:
        index = Index(vals, dtype=bool)
    else:
        index = Index(vals)
    assert type(index) is Index
    assert index.dtype == bool