import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_msg_data(self):
    zmq_msg = ffi.new('zmq_msg_t*')
    message = ffi.new('char[]', b'Hello')
    assert 0 == C.zmq_msg_init_data(zmq_msg, ffi.cast('void*', message), 5, ffi.NULL, ffi.NULL)
    data = C.zmq_msg_data(zmq_msg)
    assert ffi.NULL != zmq_msg
    assert ffi.string(ffi.cast('char*', data)) == b'Hello'
    assert 0 == C.zmq_msg_close(zmq_msg)