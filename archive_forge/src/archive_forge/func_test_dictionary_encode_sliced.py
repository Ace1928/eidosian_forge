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
def test_dictionary_encode_sliced():
    cases = [(pa.array([1, 2, 3, None, 1, 2, 3])[1:-1], pa.DictionaryArray.from_arrays(pa.array([0, 1, None, 2, 0], type='int32'), [2, 3, 1])), (pa.array([None, 'foo', 'bar', 'foo', 'xyzzy'])[1:-1], pa.DictionaryArray.from_arrays(pa.array([0, 1, 0], type='int32'), ['foo', 'bar'])), (pa.array([None, 'foo', 'bar', 'foo', 'xyzzy'], type=pa.large_string())[1:-1], pa.DictionaryArray.from_arrays(pa.array([0, 1, 0], type='int32'), pa.array(['foo', 'bar'], type=pa.large_string())))]
    for arr, expected in cases:
        result = arr.dictionary_encode()
        assert result.equals(expected)
        result = pa.chunked_array([arr]).dictionary_encode()
        assert result.num_chunks == 1
        assert result.type == expected.type
        assert result.chunk(0).equals(expected)
        result = pa.chunked_array([], type=arr.type).dictionary_encode()
        assert result.num_chunks == 0
        assert result.type == expected.type
    array = pa.array(['foo', 'bar', 'baz'])
    array.slice(1).dictionary_encode()