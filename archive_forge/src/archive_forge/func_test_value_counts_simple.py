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
def test_value_counts_simple():
    cases = [(pa.array([1, 2, 3, 1, 2, 3]), pa.array([1, 2, 3]), pa.array([2, 2, 2], type=pa.int64())), (pa.array(['foo', None, 'bar', 'foo']), pa.array(['foo', None, 'bar']), pa.array([2, 1, 1], type=pa.int64())), (pa.array(['foo', None, 'bar', 'foo'], pa.large_binary()), pa.array(['foo', None, 'bar'], pa.large_binary()), pa.array([2, 1, 1], type=pa.int64()))]
    for arr, expected_values, expected_counts in cases:
        for arr_in in (arr, pa.chunked_array([arr])):
            result = arr_in.value_counts()
            assert result.type.equals(pa.struct([pa.field('values', arr.type), pa.field('counts', pa.int64())]))
            assert result.field('values').equals(expected_values)
            assert result.field('counts').equals(expected_counts)