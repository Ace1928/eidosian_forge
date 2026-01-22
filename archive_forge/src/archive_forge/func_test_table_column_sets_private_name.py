from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_column_sets_private_name():
    t = pa.table([pa.array([1, 2, 3, 4])], names=['a0'])
    assert t[0]._name == 'a0'