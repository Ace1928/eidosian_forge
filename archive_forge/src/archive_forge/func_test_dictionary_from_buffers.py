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
@pytest.mark.parametrize('offset', (0, 1))
def test_dictionary_from_buffers(offset):
    a = pa.array(['one', 'two', 'three', 'two', 'one']).dictionary_encode()
    b = pa.DictionaryArray.from_buffers(a.type, len(a) - offset, a.indices.buffers(), a.dictionary, offset=offset)
    assert a[offset:] == b