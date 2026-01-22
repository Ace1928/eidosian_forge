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
def test_statistics_convert_logical_types(tempdir):
    cases = [(10, 11164359321221007157, pa.uint64()), (10, 4294967295, pa.uint32()), ('ähnlich', 'öffentlich', pa.utf8()), (datetime.time(10, 30, 0, 1000), datetime.time(15, 30, 0, 1000), pa.time32('ms')), (datetime.time(10, 30, 0, 1000), datetime.time(15, 30, 0, 1000), pa.time64('us')), (datetime.datetime(2019, 6, 24, 0, 0, 0, 1000), datetime.datetime(2019, 6, 25, 0, 0, 0, 1000), pa.timestamp('ms')), (datetime.datetime(2019, 6, 24, 0, 0, 0, 1000), datetime.datetime(2019, 6, 25, 0, 0, 0, 1000), pa.timestamp('us')), (datetime.date(2019, 6, 24), datetime.date(2019, 6, 25), pa.date32()), (decimal.Decimal('20.123'), decimal.Decimal('20.124'), pa.decimal128(12, 5))]
    for i, (min_val, max_val, typ) in enumerate(cases):
        t = pa.Table.from_arrays([pa.array([min_val, max_val], type=typ)], ['col'])
        path = str(tempdir / 'example{}.parquet'.format(i))
        pq.write_table(t, path, version='2.6')
        pf = pq.ParquetFile(path)
        stats = pf.metadata.row_group(0).column(0).statistics
        assert stats.min == min_val
        assert stats.max == max_val