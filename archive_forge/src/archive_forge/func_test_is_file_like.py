import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_is_file_like():

    class MockFile:
        pass
    is_file = inference.is_file_like
    data = StringIO('data')
    assert is_file(data)
    m = MockFile()
    assert not is_file(m)
    MockFile.write = lambda self: 0
    m = MockFile()
    assert not is_file(m)
    MockFile.__iter__ = lambda self: self
    m = MockFile()
    assert is_file(m)
    del MockFile.write
    MockFile.read = lambda self: 0
    m = MockFile()
    assert is_file(m)
    data = [1, 2, 3]
    assert not is_file(data)