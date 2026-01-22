from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_iterables(self):
    df1 = DataFrame([1, 2, 3])
    df2 = DataFrame([4, 5, 6])
    expected = DataFrame([1, 2, 3, 4, 5, 6])
    tm.assert_frame_equal(concat((df1, df2), ignore_index=True), expected)
    tm.assert_frame_equal(concat([df1, df2], ignore_index=True), expected)
    tm.assert_frame_equal(concat((df for df in (df1, df2)), ignore_index=True), expected)
    tm.assert_frame_equal(concat(deque((df1, df2)), ignore_index=True), expected)

    class CustomIterator1:

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index):
            try:
                return {0: df1, 1: df2}[index]
            except KeyError as err:
                raise IndexError from err
    tm.assert_frame_equal(concat(CustomIterator1(), ignore_index=True), expected)

    class CustomIterator2(abc.Iterable):

        def __iter__(self) -> Iterator:
            yield df1
            yield df2
    tm.assert_frame_equal(concat(CustomIterator2(), ignore_index=True), expected)