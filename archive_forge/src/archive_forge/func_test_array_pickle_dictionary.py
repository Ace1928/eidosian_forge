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
def test_array_pickle_dictionary(pickle_module):
    array = pa.DictionaryArray.from_arrays([0, 1, 2, 0, 1], ['a', 'b', 'c'])
    for proto in range(0, pickle_module.HIGHEST_PROTOCOL + 1):
        result = pickle_module.loads(pickle_module.dumps(array, proto))
        assert array.equals(result)