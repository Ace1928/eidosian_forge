import pytest
import pandas._testing as tm
from pandas.io.parsers import read_csv
@pytest.fixture
def df_ref(datapath):
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath = datapath('io', 'data', 'csv', 'test1.csv')
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine='python')
    return df_ref