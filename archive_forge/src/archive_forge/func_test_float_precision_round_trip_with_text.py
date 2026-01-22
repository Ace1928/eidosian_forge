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
def test_float_precision_round_trip_with_text(c_parser_only):
    parser = c_parser_only
    df = parser.read_csv(StringIO('a'), header=None, float_precision='round_trip')
    tm.assert_frame_equal(df, DataFrame({0: ['a']}))