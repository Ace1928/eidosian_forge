import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.parametrize('bits', [8, 16, 32, 64])
def test_signed_integer_overflow(bits):
    ty = getattr(pa, 'int%d' % bits)()
    with pytest.raises((OverflowError, pa.ArrowInvalid)):
        pa.array([2 ** (bits - 1)], ty)
    with pytest.raises((OverflowError, pa.ArrowInvalid)):
        pa.array([-2 ** (bits - 1) - 1], ty)