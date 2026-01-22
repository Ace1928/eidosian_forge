import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def test_chunked_categorical(version):
    df = DataFrame({'cats': Series(['a', 'b', 'a', 'b', 'c'], dtype='category')})
    df.index.name = 'index'
    expected = df.copy()
    expected.index = expected.index.astype(np.int32)
    with tm.ensure_clean() as path:
        df.to_stata(path, version=version)
        with StataReader(path, chunksize=2, order_categoricals=False) as reader:
            for i, block in enumerate(reader):
                block = block.set_index('index')
                assert 'cats' in block
                tm.assert_series_equal(block.cats, expected.cats.iloc[2 * i:2 * (i + 1)])