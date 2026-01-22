from __future__ import annotations
import decimal
import signal
import sys
import threading
import pytest
from dask.datasets import timeseries
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import assert_eq
@pytest.mark.skipif(not PANDAS_GE_150, reason='Requires pyarrow-backed nullable dtypes')
def test_read_decimal_dtype_pyarrow(spark_session, tmpdir):
    tmpdir = str(tmpdir)
    npartitions = 3
    size = 6
    decimal_data = [decimal.Decimal('8093.234'), decimal.Decimal('8094.234'), decimal.Decimal('8095.234'), decimal.Decimal('8096.234'), decimal.Decimal('8097.234'), decimal.Decimal('8098.234')]
    pdf = pd.DataFrame({'a': range(size), 'b': decimal_data})
    sdf = spark_session.createDataFrame(pdf)
    sdf = sdf.withColumn('b', sdf['b'].cast(pyspark.sql.types.DecimalType(7, 3)))
    sdf.repartition(npartitions).write.parquet(tmpdir, mode='overwrite')
    ddf = dd.read_parquet(tmpdir, engine='pyarrow', dtype_backend='pyarrow')
    assert ddf.b.dtype.pyarrow_dtype == pa.decimal128(7, 3)
    assert ddf.b.compute().dtype.pyarrow_dtype == pa.decimal128(7, 3)
    expected = pdf.astype({'a': 'int64[pyarrow]', 'b': pd.ArrowDtype(pa.decimal128(7, 3))})
    assert_eq(ddf, expected, check_index=False)