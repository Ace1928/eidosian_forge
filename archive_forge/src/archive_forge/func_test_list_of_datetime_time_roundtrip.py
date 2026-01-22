import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_list_of_datetime_time_roundtrip():
    times = pd.to_datetime(['09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00'], format='%H:%M')
    df = pd.DataFrame({'time': [times.time]})
    _roundtrip_pandas_dataframe(df, write_kwargs={})