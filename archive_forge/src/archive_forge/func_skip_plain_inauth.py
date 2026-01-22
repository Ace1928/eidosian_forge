import contextlib
import os
import time
from threading import Thread
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
from zmq.utils import z85
def skip_plain_inauth(self):
    """test PLAIN failed authentication"""
    server = self.socket(zmq.DEALER)
    server.identity = b'IDENT'
    client = self.socket(zmq.DEALER)
    self.sockets.extend([server, client])
    client.plain_username = USER
    client.plain_password = b'incorrect'
    server.plain_server = True
    assert server.mechanism == zmq.PLAIN
    assert client.mechanism == zmq.PLAIN
    with self.zap():
        iface = 'tcp://127.0.0.1'
        port = server.bind_to_random_port(iface)
        client.connect('%s:%i' % (iface, port))
        client.send(b'ping')
        server.rcvtimeo = 250
        self.assertRaisesErrno(zmq.EAGAIN, server.recv)