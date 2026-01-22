import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_fillna_dict_series():
    frame_data = {'a': [np.nan, 1, 2, np.nan, np.nan], 'b': [1, 2, 3, np.nan, np.nan], 'c': [np.nan, 1, 2, 3, 4]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df_equals(modin_df.fillna({'a': 0, 'b': 5}), df.fillna({'a': 0, 'b': 5}))
    df_equals(modin_df.fillna({'a': 0, 'b': 5, 'd': 7}), df.fillna({'a': 0, 'b': 5, 'd': 7}))
    df_equals(modin_df.fillna(modin_df.max()), df.fillna(df.max()))