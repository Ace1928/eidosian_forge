import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def test_read_tag():
    str_io = BytesIO()
    r = _make_readerlike(str_io)
    c_reader = m5u.VarReader5(r)
    assert_raises(OSError, c_reader.read_tag)
    tag = _make_tag('i4', 1, mio5p.miINT32, sde=True)
    tag['byte_count'] = 5
    _write_stream(str_io, tag.tobytes())
    assert_raises(ValueError, c_reader.read_tag)