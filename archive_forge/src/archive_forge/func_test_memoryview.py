import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_memoryview(self):
    """test messages from memoryview"""
    s = b'carrotjuice'
    memoryview(s)
    m = zmq.Frame(s)
    buf = m.buffer
    s2 = buf.tobytes()
    assert s2 == s
    assert m.bytes == s