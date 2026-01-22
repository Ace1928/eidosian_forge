import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_msg_init_close(self):
    zmq_msg = ffi.new('zmq_msg_t*')
    assert ffi.NULL != zmq_msg
    assert 0 == C.zmq_msg_init(zmq_msg)
    assert 0 == C.zmq_msg_close(zmq_msg)