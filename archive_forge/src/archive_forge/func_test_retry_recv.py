import signal
import time
from threading import Thread
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, SkipTest
@mark.skipif(not hasattr(zmq, 'RCVTIMEO'), reason='requires RCVTIMEO')
def test_retry_recv(self):
    pull = self.socket(zmq.PULL)
    pull.rcvtimeo = self.timeout_ms
    self.alarm()
    self.assertRaises(zmq.Again, pull.recv)
    assert self.timer_fired