import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
@skip_pypy
def test_lifecycle1(self):
    """Run through a ref counting cycle with a copy."""
    for i in range(5, 16):
        s = 2 ** i * x
        rc = rc_0 = grc(s)
        m = zmq.Frame(s, copy=False)
        rc += 2
        assert grc(s) == rc
        m2 = copy.copy(m)
        rc += 1
        assert grc(s) == rc
        buf = m2.buffer
        assert grc(s) == rc
        assert s == bytes(m2)
        assert s == m.bytes
        assert s == bytes(buf)
        del m2
        assert grc(s) == rc
        del buf
        rc -= 1
        assert grc(s) == rc
        del m
        rc -= 2
        await_gc(s, rc)
        assert grc(s) == rc
        assert rc == rc_0
        del s