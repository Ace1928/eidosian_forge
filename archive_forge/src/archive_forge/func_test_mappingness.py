import logging
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError
def test_mappingness(self):
    hdr = self.header_class()
    with pytest.raises(ValueError):
        hdr['nonexistent key'] = 0.1
    hdr_dt = hdr.structarr.dtype
    keys = hdr.keys()
    assert keys == list(hdr)
    vals = hdr.values()
    assert len(vals) == len(keys)
    assert keys == list(hdr_dt.names)
    for key, val in hdr.items():
        assert_array_equal(hdr[key], val)
    assert hdr.get('nonexistent key') is None
    assert hdr.get('nonexistent key', 'default') == 'default'
    assert hdr.get(keys[0]) == vals[0]
    assert hdr.get(keys[0], 'default') == vals[0]
    falsyval = 0 if np.issubdtype(hdr_dt[0], np.number) else b''
    hdr[keys[0]] = falsyval
    assert hdr[keys[0]] == falsyval
    assert hdr.get(keys[0]) == falsyval
    assert hdr.get(keys[0], -1) == falsyval