from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_axis(education_df):
    gp = education_df.groupby('country', axis=1)
    with pytest.raises(NotImplementedError, match='axis'):
        gp.value_counts()