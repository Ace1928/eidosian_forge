import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
def test_pubsub(self):
    s1, s2 = self.create_bound_pair(zmq.PUB, zmq.SUB)
    s2.setsockopt(zmq.SUBSCRIBE, b'')
    wait()
    poller = self.Poller()
    poller.register(s1, zmq.POLLIN | zmq.POLLOUT)
    poller.register(s2, zmq.POLLIN)
    socks = dict(poller.poll())
    assert socks[s1] == zmq.POLLOUT
    assert s2 not in socks
    s1.send(b'msg1')
    socks = dict(poller.poll())
    assert socks[s1] == zmq.POLLOUT
    wait()
    socks = dict(poller.poll())
    assert socks[s2] == zmq.POLLIN
    s2.recv()
    socks = dict(poller.poll())
    assert s2 not in socks
    poller.unregister(s1)
    poller.unregister(s2)