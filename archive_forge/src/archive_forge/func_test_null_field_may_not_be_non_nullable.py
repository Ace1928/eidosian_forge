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
def test_null_field_may_not_be_non_nullable():
    with pytest.raises(ValueError):
        pa.field('f0', pa.null(), nullable=False)