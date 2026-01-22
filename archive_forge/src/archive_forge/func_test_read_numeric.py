import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def test_read_numeric():
    str_io = BytesIO()
    r = _make_readerlike(str_io)
    for base_dt, val, mdtype in (('u2', 30, mio5p.miUINT16), ('i4', 1, mio5p.miINT32), ('i2', -1, mio5p.miINT16)):
        for byte_code in ('<', '>'):
            r.byte_order = byte_code
            c_reader = m5u.VarReader5(r)
            assert_equal(c_reader.little_endian, byte_code == '<')
            assert_equal(c_reader.is_swapped, byte_code != boc.native_code)
            for sde_f in (False, True):
                dt = np.dtype(base_dt).newbyteorder(byte_code)
                a = _make_tag(dt, val, mdtype, sde_f)
                a_str = a.tobytes()
                _write_stream(str_io, a_str)
                el = c_reader.read_numeric()
                assert_equal(el, val)
                _write_stream(str_io, a_str, a_str)
                el = c_reader.read_numeric()
                assert_equal(el, val)
                el = c_reader.read_numeric()
                assert_equal(el, val)