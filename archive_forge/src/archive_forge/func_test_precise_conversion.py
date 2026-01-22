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
@td.skip_if_32bit
@pytest.mark.slow
@pytest.mark.parametrize('num', np.linspace(1.0, 2.0, num=21))
def test_precise_conversion(c_parser_only, num):
    parser = c_parser_only
    normal_errors = []
    precise_errors = []

    def error(val: float, actual_val: Decimal) -> Decimal:
        return abs(Decimal(f'{val:.100}') - actual_val)
    text = f'a\n{num:.25}'
    normal_val = float(parser.read_csv(StringIO(text), float_precision='legacy')['a'][0])
    precise_val = float(parser.read_csv(StringIO(text), float_precision='high')['a'][0])
    roundtrip_val = float(parser.read_csv(StringIO(text), float_precision='round_trip')['a'][0])
    actual_val = Decimal(text[2:])
    normal_errors.append(error(normal_val, actual_val))
    precise_errors.append(error(precise_val, actual_val))
    assert roundtrip_val == float(text[2:])
    assert sum(precise_errors) <= sum(normal_errors)
    assert max(precise_errors) <= max(normal_errors)