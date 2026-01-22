import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isna_behavior(idx):
    msg = 'isna is not defined for MultiIndex'
    with pytest.raises(NotImplementedError, match=msg):
        pd.isna(idx)