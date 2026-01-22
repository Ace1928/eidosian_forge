import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.pandas
def test_parquet_raise_on_unset_statistics():
    df = pd.DataFrame({'t': pd.Series([pd.NaT], dtype='datetime64[ns]')})
    meta = make_sample_file(pa.Table.from_pandas(df)).metadata
    assert not meta.row_group(0).column(0).statistics.has_min_max
    assert meta.row_group(0).column(0).statistics.max is None