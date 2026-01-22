import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
@pytest.mark.skipif(sys.platform.startswith('win'), reason='Windows does not support polling on files')
def test_poll_raw(self):

    async def test():
        p = future.Poller()
        r, w = os.pipe()
        r = os.fdopen(r, 'rb')
        w = os.fdopen(w, 'wb')
        p.register(r, zmq.POLLIN)
        p.register(w, zmq.POLLOUT)
        evts = await p.poll(timeout=1)
        evts = dict(evts)
        assert r.fileno() not in evts
        assert w.fileno() in evts
        assert evts[w.fileno()] == zmq.POLLOUT
        p.unregister(w)
        w.write(b'x')
        w.flush()
        evts = await p.poll(timeout=1000)
        evts = dict(evts)
        assert r.fileno() in evts
        assert evts[r.fileno()] == zmq.POLLIN
        assert r.read(1) == b'x'
        r.close()
        w.close()
    self.loop.run_sync(test)