from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
@pytest.mark.pandas
def test_min_chunksize():
    data = pd.DataFrame([np.arange(4)], columns=['A', 'B', 'C', 'D'])
    table = pa.Table.from_pandas(data.reset_index())
    buf = io.BytesIO()
    _write_table(table, buf, chunk_size=-1)
    buf.seek(0)
    result = _read_table(buf)
    assert result.equals(table)
    with pytest.raises(ValueError):
        _write_table(table, buf, chunk_size=0)