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
@pytest.mark.parametrize('write_index', [True, False])
def test_value_labels_iterator(self, write_index):
    d = {'A': ['B', 'E', 'C', 'A', 'E']}
    df = DataFrame(data=d)
    df['A'] = df['A'].astype('category')
    with tm.ensure_clean() as path:
        df.to_stata(path, write_index=write_index)
        with read_stata(path, iterator=True) as dta_iter:
            value_labels = dta_iter.value_labels()
    assert value_labels == {'A': {0: 'A', 1: 'B', 2: 'C', 3: 'E'}}