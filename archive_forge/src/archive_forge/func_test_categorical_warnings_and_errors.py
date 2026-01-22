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
def test_categorical_warnings_and_errors(self):
    original = DataFrame.from_records([['a' * 10000], ['b' * 10000], ['c' * 10000], ['d' * 10000]], columns=['Too_long'])
    original = pd.concat([original[col].astype('category') for col in original], axis=1)
    with tm.ensure_clean() as path:
        msg = 'Stata value labels for a single variable must have a combined length less than 32,000 characters\\.'
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path)
    original = DataFrame.from_records([['a'], ['b'], ['c'], ['d'], [1]], columns=['Too_long'])
    original = pd.concat([original[col].astype('category') for col in original], axis=1)
    with tm.assert_produces_warning(ValueLabelTypeMismatch):
        original.to_stata(path)