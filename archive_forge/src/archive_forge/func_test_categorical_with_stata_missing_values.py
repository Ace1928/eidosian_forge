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
def test_categorical_with_stata_missing_values(self, version):
    values = [['a' + str(i)] for i in range(120)]
    values.append([np.nan])
    original = DataFrame.from_records(values, columns=['many_labels'])
    original = pd.concat([original[col].astype('category') for col in original], axis=1)
    original.index.name = 'index'
    with tm.ensure_clean() as path:
        original.to_stata(path, version=version)
        written_and_read_again = self.read_dta(path)
    res = written_and_read_again.set_index('index')
    expected = original.copy()
    for col in expected:
        cat = expected[col]._values
        new_cats = cat.remove_unused_categories().categories
        cat = cat.set_categories(new_cats, ordered=True)
        expected[col] = cat
    expected.index = expected.index.astype(np.int32)
    tm.assert_frame_equal(res, expected)