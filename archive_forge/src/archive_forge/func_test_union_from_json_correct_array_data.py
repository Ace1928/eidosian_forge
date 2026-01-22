import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_union_from_json_correct_array_data():
    raw_data = memoryview(np.zeros((4, 3), dtype=np.float32))
    json_data = {'buffer': raw_data, 'dtype': 'float32', 'shape': [4, 3]}
    data = data_union_from_json(json_data, None)
    np.testing.assert_equal(raw_data, data)