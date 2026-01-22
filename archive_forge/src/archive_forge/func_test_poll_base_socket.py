import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
@pytest.mark.skipif(sys.platform.startswith('win'), reason='Windows unsupported socket type')
def test_poll_base_socket(self):

    async def test():
        ctx = zmq.Context()
        url = 'inproc://test'
        a = ctx.socket(zmq.PUSH)
        b = ctx.socket(zmq.PULL)
        self.sockets.extend([a, b])
        a.bind(url)
        b.connect(url)
        poller = future.Poller()
        poller.register(b, zmq.POLLIN)
        f = poller.poll(timeout=1000)
        assert not f.done()
        a.send_multipart([b'hi', b'there'])
        evt = await f
        assert evt == [(b, zmq.POLLIN)]
        recvd = b.recv_multipart()
        assert recvd == [b'hi', b'there']
        a.close()
        b.close()
        ctx.term()
    self.loop.run_sync(test)