import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest, have_gevent
def test_green_device(self):
    rep = self.context.socket(zmq.REP)
    req = self.context.socket(zmq.REQ)
    self.sockets.extend([req, rep])
    port = rep.bind_to_random_port('tcp://127.0.0.1')
    g = gevent.spawn(zmq.green.device, zmq.QUEUE, rep, rep)
    req.connect('tcp://127.0.0.1:%i' % port)
    req.send(b'hi')
    timeout = gevent.Timeout(3)
    timeout.start()
    receiver = gevent.spawn(req.recv)
    assert receiver.get(2) == b'hi'
    timeout.cancel()
    g.kill(block=True)