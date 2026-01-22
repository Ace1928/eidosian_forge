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
def test_to_csv_stdout_file(self, capsys):
    df = DataFrame([['foo', 'bar'], ['baz', 'qux']], columns=['name_1', 'name_2'])
    expected_rows = [',name_1,name_2', '0,foo,bar', '1,baz,qux']
    expected_ascii = tm.convert_rows_list_to_csv_str(expected_rows)
    df.to_csv(sys.stdout, encoding='ascii')
    captured = capsys.readouterr()
    assert captured.out == expected_ascii
    assert not sys.stdout.closed