from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_dunder_init():
    with pytest.raises(TypeError, match='RecordBatch'):
        pa.RecordBatch()