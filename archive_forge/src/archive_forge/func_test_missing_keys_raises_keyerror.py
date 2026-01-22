import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_missing_keys_raises_keyerror(self):
    df = DataFrame(np.arange(12).reshape(4, 3), columns=['A', 'B', 'C'])
    df2 = df.set_index(['A', 'B'])
    with pytest.raises(KeyError, match='1'):
        df2.loc[1, 6]