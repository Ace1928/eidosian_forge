import pytest
import numpy as np
import zlib
from traitlets import HasTraits, Instance, Undefined
from ipywidgets import Widget, widget_serialization
from ..ndarray.union import DataUnion
from ..ndarray.serializers import (
def test_array_to_json_int64_warning():
    data = np.zeros((4, 3), dtype=np.uint64, order='F')
    with pytest.warns(UserWarning) as captured_warnings:
        json_data = array_to_json(data, None)
        assert len(captured_warnings) == 1
        assert 'Cannot serialize (u)int64 data' in str(captured_warnings[0].message)
    reinterpreted_data = array_from_json(json_data, None)
    np.testing.assert_equal(data, reinterpreted_data)
    assert reinterpreted_data.flags['C_CONTIGUOUS']
    assert reinterpreted_data.dtype == np.uint32