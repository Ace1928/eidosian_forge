import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_msg_init_data(self):
    zmq_msg = ffi.new('zmq_msg_t*')
    message = ffi.new('char[5]', b'Hello')
    assert 0 == C.zmq_msg_init_data(zmq_msg, ffi.cast('void*', message), 5, ffi.NULL, ffi.NULL)
    assert ffi.NULL != zmq_msg
    assert 0 == C.zmq_msg_close(zmq_msg)