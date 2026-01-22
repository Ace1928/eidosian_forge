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
def test_iterator_value_labels():
    values = ['c_label', 'b_label'] + ['a_label'] * 500
    df = DataFrame({f'col{k}': pd.Categorical(values, ordered=True) for k in range(2)})
    with tm.ensure_clean() as path:
        df.to_stata(path, write_index=False)
        expected = pd.Index(['a_label', 'b_label', 'c_label'], dtype='object')
        with read_stata(path, chunksize=100) as reader:
            for j, chunk in enumerate(reader):
                for i in range(2):
                    tm.assert_index_equal(chunk.dtypes.iloc[i].categories, expected)
                tm.assert_frame_equal(chunk, df.iloc[j * 100:(j + 1) * 100])