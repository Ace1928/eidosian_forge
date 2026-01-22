import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_no_second_level_index(self):
    df = DataFrame(index=MultiIndex.from_product([list('ab'), list('cd'), list('e')]), columns=['Val'])
    res = df.loc[np.s_[:, 'c', :]]
    expected = DataFrame(index=MultiIndex.from_product([list('ab'), list('e')]), columns=['Val'])
    tm.assert_frame_equal(res, expected)