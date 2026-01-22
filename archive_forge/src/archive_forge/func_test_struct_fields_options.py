from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_struct_fields_options():
    a = pa.array([4, 5, 6], type=pa.int64())
    b = pa.array(['bar', None, ''])
    c = pa.StructArray.from_arrays([a, b], ['a', 'b'])
    arr = pa.StructArray.from_arrays([a, c], ['a', 'c'])
    assert pc.struct_field(arr, '.c.b') == b
    assert pc.struct_field(arr, b'.c.b') == b
    assert pc.struct_field(arr, ['c', 'b']) == b
    assert pc.struct_field(arr, [1, 'b']) == b
    assert pc.struct_field(arr, (b'c', 'b')) == b
    assert pc.struct_field(arr, pc.field(('c', 'b'))) == b
    assert pc.struct_field(arr, '.a') == a
    assert pc.struct_field(arr, ['a']) == a
    assert pc.struct_field(arr, 'a') == a
    assert pc.struct_field(arr, pc.field(('a',))) == a
    assert pc.struct_field(arr, indices=[1, 1]) == b
    assert pc.struct_field(arr, (1, 1)) == b
    assert pc.struct_field(arr, [0]) == a
    assert pc.struct_field(arr, []) == arr
    with pytest.raises(pa.ArrowInvalid, match='No match for FieldRef'):
        pc.struct_field(arr, 'foo')
    with pytest.raises(pa.ArrowInvalid, match='No match for FieldRef'):
        pc.struct_field(arr, '.c.foo')
    with pytest.raises(pa.ArrowInvalid, match='No match for FieldRef'):
        pc.struct_field(arr, '.a.foo')