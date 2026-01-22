from threading import Thread
import zmq
from zmq import Again, ContextTerminated, ZMQError, strerror
from zmq.tests import BaseZMQTestCase
def test_again(self):
    s = self.context.socket(zmq.REP)
    self.assertRaises(Again, s.recv, zmq.NOBLOCK)
    self.assertRaisesErrno(zmq.EAGAIN, s.recv, zmq.NOBLOCK)
    s.close()