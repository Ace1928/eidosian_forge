import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_na_position_with_categories_raises(self):
    df = DataFrame({'c': Categorical(['A', np.nan, 'B', np.nan, 'C'], categories=['A', 'B', 'C'], ordered=True)})
    with pytest.raises(ValueError, match='invalid na_position: bad_position'):
        df.sort_values(by='c', ascending=False, na_position='bad_position')