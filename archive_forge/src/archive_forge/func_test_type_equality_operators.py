from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_type_equality_operators():
    many_types = get_many_types()
    non_pyarrow = ('foo', 16, {'s', 'e', 't'})
    for index, ty in enumerate(many_types):
        for i, other in enumerate(many_types + non_pyarrow):
            if i == index:
                assert ty == other
            else:
                assert ty != other