import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def test_read_stream():
    tag = _make_tag('i4', 1, mio5p.miINT32, sde=True)
    tag_str = tag.tobytes()
    str_io = BytesIO(tag_str)
    st = streams.make_stream(str_io)
    s = streams._read_into(st, tag.itemsize)
    assert_equal(s, tag.tobytes())