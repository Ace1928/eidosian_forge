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
def test_from_records_tuples_generator(self):

    def tuple_generator(length):
        for i in range(length):
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            yield (i, letters[i % len(letters)], i / length)
    columns_names = ['Integer', 'String', 'Float']
    columns = [[i[j] for i in tuple_generator(10)] for j in range(len(columns_names))]
    data = {'Integer': columns[0], 'String': columns[1], 'Float': columns[2]}
    expected = DataFrame(data, columns=columns_names)
    generator = tuple_generator(10)
    result = DataFrame.from_records(generator, columns=columns_names)
    tm.assert_frame_equal(result, expected)