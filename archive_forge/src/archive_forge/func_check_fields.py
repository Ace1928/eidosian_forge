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
def check_fields(ty, fields):
    assert ty.num_fields == len(fields)
    assert [ty[i] for i in range(ty.num_fields)] == fields
    assert [ty.field(i) for i in range(ty.num_fields)] == fields