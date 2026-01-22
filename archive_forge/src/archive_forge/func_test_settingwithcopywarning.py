import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def test_settingwithcopywarning(self):
    """Test that MICEData does not throw a SettingWithCopyWarning when imputing (https://github.com/statsmodels/statsmodels/issues/5430)"""
    df = gendat()
    df['intcol'] = np.arange(len(df))
    df['intcol'] = df.intcol.astype('int32')
    miceData = mice.MICEData(df)
    with pd.option_context('mode.chained_assignment', 'warn'):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            miceData.update_all()
            ws = [w for w in ws if '\\pandas\\' in w.filename]
            assert len(ws) == 0