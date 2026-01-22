import datetime
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
@pytest.mark.filterwarnings('ignore:ChainedAssignmentError:FutureWarning')
@pytest.mark.parametrize('path_klass', [lambda p: p, Path])
def test_spss_labelled_num(path_klass, datapath):
    fname = path_klass(datapath('io', 'data', 'spss', 'labelled-num.sav'))
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({'VAR00002': 'This is one'}, index=[0])
    expected['VAR00002'] = pd.Categorical(expected['VAR00002'])
    tm.assert_frame_equal(df, expected)
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({'VAR00002': 1.0}, index=[0])
    tm.assert_frame_equal(df, expected)