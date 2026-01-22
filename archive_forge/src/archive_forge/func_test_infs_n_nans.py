from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('grps', [['qux'], ['qux', 'quux']])
@pytest.mark.parametrize('vals', [[-np.inf, -np.inf, np.nan, 1.0, np.nan, np.inf, np.inf]])
@pytest.mark.parametrize('ties_method,ascending,na_option,exp', [('average', True, 'keep', [1.5, 1.5, np.nan, 3, np.nan, 4.5, 4.5]), ('average', True, 'top', [3.5, 3.5, 1.5, 5.0, 1.5, 6.5, 6.5]), ('average', True, 'bottom', [1.5, 1.5, 6.5, 3.0, 6.5, 4.5, 4.5]), ('average', False, 'keep', [4.5, 4.5, np.nan, 3, np.nan, 1.5, 1.5]), ('average', False, 'top', [6.5, 6.5, 1.5, 5.0, 1.5, 3.5, 3.5]), ('average', False, 'bottom', [4.5, 4.5, 6.5, 3.0, 6.5, 1.5, 1.5]), ('min', True, 'keep', [1.0, 1.0, np.nan, 3.0, np.nan, 4.0, 4.0]), ('min', True, 'top', [3.0, 3.0, 1.0, 5.0, 1.0, 6.0, 6.0]), ('min', True, 'bottom', [1.0, 1.0, 6.0, 3.0, 6.0, 4.0, 4.0]), ('min', False, 'keep', [4.0, 4.0, np.nan, 3.0, np.nan, 1.0, 1.0]), ('min', False, 'top', [6.0, 6.0, 1.0, 5.0, 1.0, 3.0, 3.0]), ('min', False, 'bottom', [4.0, 4.0, 6.0, 3.0, 6.0, 1.0, 1.0]), ('max', True, 'keep', [2.0, 2.0, np.nan, 3.0, np.nan, 5.0, 5.0]), ('max', True, 'top', [4.0, 4.0, 2.0, 5.0, 2.0, 7.0, 7.0]), ('max', True, 'bottom', [2.0, 2.0, 7.0, 3.0, 7.0, 5.0, 5.0]), ('max', False, 'keep', [5.0, 5.0, np.nan, 3.0, np.nan, 2.0, 2.0]), ('max', False, 'top', [7.0, 7.0, 2.0, 5.0, 2.0, 4.0, 4.0]), ('max', False, 'bottom', [5.0, 5.0, 7.0, 3.0, 7.0, 2.0, 2.0]), ('first', True, 'keep', [1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0]), ('first', True, 'top', [3.0, 4.0, 1.0, 5.0, 2.0, 6.0, 7.0]), ('first', True, 'bottom', [1.0, 2.0, 6.0, 3.0, 7.0, 4.0, 5.0]), ('first', False, 'keep', [4.0, 5.0, np.nan, 3.0, np.nan, 1.0, 2.0]), ('first', False, 'top', [6.0, 7.0, 1.0, 5.0, 2.0, 3.0, 4.0]), ('first', False, 'bottom', [4.0, 5.0, 6.0, 3.0, 7.0, 1.0, 2.0]), ('dense', True, 'keep', [1.0, 1.0, np.nan, 2.0, np.nan, 3.0, 3.0]), ('dense', True, 'top', [2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 4.0]), ('dense', True, 'bottom', [1.0, 1.0, 4.0, 2.0, 4.0, 3.0, 3.0]), ('dense', False, 'keep', [3.0, 3.0, np.nan, 2.0, np.nan, 1.0, 1.0]), ('dense', False, 'top', [4.0, 4.0, 1.0, 3.0, 1.0, 2.0, 2.0]), ('dense', False, 'bottom', [3.0, 3.0, 4.0, 2.0, 4.0, 1.0, 1.0])])
def test_infs_n_nans(grps, vals, ties_method, ascending, na_option, exp):
    key = np.repeat(grps, len(vals))
    vals = vals * len(grps)
    df = DataFrame({'key': key, 'val': vals})
    result = df.groupby('key').rank(method=ties_method, ascending=ascending, na_option=na_option)
    exp_df = DataFrame(exp * len(grps), columns=['val'])
    tm.assert_frame_equal(result, exp_df)