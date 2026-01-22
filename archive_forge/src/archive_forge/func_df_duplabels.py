import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
@pytest.fixture
def df_duplabels(df):
    """DataFrame with level 'L1' and labels 'L2', 'L3', and 'L2'"""
    df = df.set_index(['L1'])
    df = pd.concat([df, df['L2']], axis=1)
    return df