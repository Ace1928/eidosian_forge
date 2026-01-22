from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_usecols_dtypes(c_parser_only):
    parser = c_parser_only
    data = '1,2,3\n4,5,6\n7,8,9\n10,11,12'
    result = parser.read_csv(StringIO(data), usecols=(0, 1, 2), names=('a', 'b', 'c'), header=None, converters={'a': str}, dtype={'b': int, 'c': float})
    result2 = parser.read_csv(StringIO(data), usecols=(0, 2), names=('a', 'b', 'c'), header=None, converters={'a': str}, dtype={'b': int, 'c': float})
    assert (result.dtypes == [object, int, float]).all()
    assert (result2.dtypes == [object, float]).all()