from threading import Thread
import zmq
from zmq import Again, ContextTerminated, ZMQError, strerror
from zmq.tests import BaseZMQTestCase
def atest_ctxterm(self):
    s = self.context.socket(zmq.REP)
    t = Thread(target=self.context.term)
    t.start()
    self.assertRaises(ContextTerminated, s.recv, zmq.NOBLOCK)
    self.assertRaisesErrno(zmq.TERM, s.recv, zmq.NOBLOCK)
    s.close()
    t.join()