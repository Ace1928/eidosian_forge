from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('allow_duplicates', ['bad value'])
def test_reset_index_allow_duplicates_check(self, multiindex_df, allow_duplicates):
    with pytest.raises(ValueError, match='expected type bool'):
        multiindex_df.reset_index(allow_duplicates=allow_duplicates)