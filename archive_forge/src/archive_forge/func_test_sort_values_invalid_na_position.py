from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.parametrize('na_position', [None, 'middle'])
def test_sort_values_invalid_na_position(index_with_missing, na_position):
    with pytest.raises(ValueError, match=f'invalid na_position: {na_position}'):
        index_with_missing.sort_values(na_position=na_position)