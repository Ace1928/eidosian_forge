from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('do_ref', [True, False])
def test_setitem_cache_updating(self, do_ref):
    cont = ['one', 'two', 'three', 'four', 'five', 'six', 'seven']
    df = DataFrame({'a': cont, 'b': cont[3:] + cont[:3], 'c': np.arange(7)})
    if do_ref:
        df.loc[0, 'c']
    df.loc[7, 'c'] = 1
    assert df.loc[0, 'c'] == 0.0
    assert df.loc[7, 'c'] == 1.0