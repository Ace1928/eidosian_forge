from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def list_generator(length):
    for i in range(length):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        yield [i, letters[i % len(letters)], i / length]