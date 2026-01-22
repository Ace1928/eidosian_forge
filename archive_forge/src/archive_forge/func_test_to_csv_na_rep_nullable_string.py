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
def test_to_csv_na_rep_nullable_string(self, nullable_string_dtype):
    expected = tm.convert_rows_list_to_csv_str([',0', '0,a', '1,ZZZZZ', '2,c'])
    csv = pd.Series(['a', pd.NA, 'c'], dtype=nullable_string_dtype).to_csv(na_rep='ZZZZZ')
    assert expected == csv