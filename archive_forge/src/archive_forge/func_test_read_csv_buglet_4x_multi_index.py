from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_read_csv_buglet_4x_multi_index(python_parser_only):
    data = '                      A       B       C       D        E\none two three   four\na   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640\na   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744\nx   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838'
    parser = python_parser_only
    expected = DataFrame([[-0.5109, -2.3358, -0.4645, 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744], [-0.6662, -0.5243, -0.358, 0.89145, 2.5838]], columns=['A', 'B', 'C', 'D', 'E'], index=MultiIndex.from_tuples([('a', 'b', 10.0032, 5), ('a', 'q', 20, 4), ('x', 'q', 30, 3)], names=['one', 'two', 'three', 'four']))
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)