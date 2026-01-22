from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_from_arrays_validate_lengths():
    data = [pa.array([1]), pa.array(['tokyo', 'like', 'happy']), pa.array(['derek'])]
    with pytest.raises(ValueError):
        pa.record_batch(data, ['id', 'tags', 'name'])