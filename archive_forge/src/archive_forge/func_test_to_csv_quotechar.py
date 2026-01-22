import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_csv_quotechar(self):
    df = DataFrame({'col': [1, 2]})
    expected = '"","col"\n"0","1"\n"1","2"\n'
    with tm.ensure_clean('test.csv') as path:
        df.to_csv(path, quoting=1)
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected
    expected = '$$,$col$\n$0$,$1$\n$1$,$2$\n'
    with tm.ensure_clean('test.csv') as path:
        df.to_csv(path, quoting=1, quotechar='$')
        with open(path, encoding='utf-8') as f:
            assert f.read() == expected
    with tm.ensure_clean('test.csv') as path:
        with pytest.raises(TypeError, match='quotechar'):
            df.to_csv(path, quoting=1, quotechar=None)