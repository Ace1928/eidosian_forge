import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_multiindex_missing_label_raises(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])
    with pytest.raises(KeyError, match='^2$'):
        df.loc[2]