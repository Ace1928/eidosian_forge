import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_frame_pad_backfill_limit(data):
    pandas_df = pandas.DataFrame(data)
    index = pandas_df.index
    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(modin_df.fillna(method='pad', limit=2), result.fillna(method='pad', limit=2))
    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(modin_df.fillna(method='backfill', limit=2), result.fillna(method='backfill', limit=2))