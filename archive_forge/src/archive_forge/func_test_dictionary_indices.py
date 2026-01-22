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
def test_dictionary_indices():
    indices = pa.array([0, 1, 2, 0, 1, 2])
    dictionary = pa.array(['foo', 'bar', 'baz'])
    arr = pa.DictionaryArray.from_arrays(indices, dictionary)
    arr.indices.validate(full=True)