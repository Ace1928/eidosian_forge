from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_column_sets_private_name():
    rb = pa.record_batch([pa.array([1, 2, 3, 4])], names=['a0'])
    assert rb[0]._name == 'a0'