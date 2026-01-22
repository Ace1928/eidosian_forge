import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
@pytest.mark.pandas
def test_feather_datetime_resolution_arrow_to_pandas(tempdir):
    from datetime import datetime
    df = pd.DataFrame({'date': [datetime.fromisoformat('1654-01-01'), datetime.fromisoformat('1920-01-01')]})
    write_feather(df, tempdir / 'test_resolution.feather')
    expected_0 = datetime.fromisoformat('1654-01-01')
    expected_1 = datetime.fromisoformat('1920-01-01')
    result = read_feather(tempdir / 'test_resolution.feather', timestamp_as_object=True)
    assert expected_0 == result['date'][0]
    assert expected_1 == result['date'][1]