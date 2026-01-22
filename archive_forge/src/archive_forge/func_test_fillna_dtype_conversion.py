import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_fillna_dtype_conversion():
    df = pandas.DataFrame(index=range(3), columns=['A', 'B'], dtype='float64')
    modin_df = pd.DataFrame(index=range(3), columns=['A', 'B'], dtype='float64')
    df_equals(modin_df.fillna('nan'), df.fillna('nan'))
    frame_data = {'A': [1, np.nan], 'B': [1.0, 2.0]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    for v in ['', 1, np.nan, 1.0]:
        df_equals(modin_df.fillna(v), df.fillna(v))