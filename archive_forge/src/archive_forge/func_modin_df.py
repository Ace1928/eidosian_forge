import pickle
import numpy as np
import pytest
import modin.pandas as pd
from modin.config import PersistentPickle
from modin.tests.pandas.utils import create_test_dfs, df_equals
@pytest.fixture
def modin_df():
    return pd.DataFrame({'col1': np.arange(1000), 'col2': np.arange(2000, 3000)})