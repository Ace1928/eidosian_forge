import signal
import time
from threading import Thread
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, SkipTest
def test_retry_term(self):
    push = self.socket(zmq.PUSH)
    push.linger = self.timeout_ms
    push.connect('tcp://127.0.0.1:5555')
    push.send(b'ping')
    time.sleep(0.1)
    self.alarm()
    self.context.destroy()
    assert self.timer_fired
    assert self.context.closed