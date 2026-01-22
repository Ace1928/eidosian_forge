from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_factory_functions_invalid_input():
    with pytest.raises(TypeError, match='Expected pandas DataFrame, python'):
        pa.table('invalid input')
    with pytest.raises(TypeError, match='Expected pandas DataFrame'):
        pa.record_batch('invalid input')