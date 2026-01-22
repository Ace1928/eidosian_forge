from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_decimal_to_int_safe():
    safe_cases = [([decimal.Decimal('123456'), None, decimal.Decimal('-912345')], pa.decimal128(32, 5), [123456, None, -912345], pa.int32()), ([decimal.Decimal('1234'), None, decimal.Decimal('-9123')], pa.decimal128(19, 10), [1234, None, -9123], pa.int16()), ([decimal.Decimal('123'), None, decimal.Decimal('-91')], pa.decimal128(19, 10), [123, None, -91], pa.int8())]
    for case in safe_cases:
        _check_cast_case(case)
        _check_cast_case(case, safe=True)