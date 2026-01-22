import pickle
import numpy as np
import pytest
import modin.pandas as pd
from modin.config import PersistentPickle
from modin.tests.pandas.utils import create_test_dfs, df_equals
@pytest.fixture
def modin_column(modin_df):
    return modin_df['col1']