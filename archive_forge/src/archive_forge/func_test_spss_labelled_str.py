import datetime
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.filterwarnings('ignore::pandas.errors.ChainedAssignmentError')
@pytest.mark.filterwarnings('ignore:ChainedAssignmentError:FutureWarning')
def test_spss_labelled_str(datapath):
    fname = datapath('io', 'data', 'spss', 'labelled-str.sav')
    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({'gender': ['Male', 'Female']})
    expected['gender'] = pd.Categorical(expected['gender'])
    tm.assert_frame_equal(df, expected)
    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({'gender': ['M', 'F']})
    tm.assert_frame_equal(df, expected)