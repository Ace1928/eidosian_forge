import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_recv_json_cancelled(self):

    async def test():
        a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
        f = b.recv_json()
        assert not f.done()
        f.cancel()
        await gen.sleep(0)
        obj = dict(a=5)
        await a.send_json(obj)
        with pytest.raises(future.CancelledError):
            recvd = await f
        assert f.done()
        events = await b.poll(timeout=5)
        assert events
        await gen.sleep(0)
        recvd = await gen.with_timeout(timedelta(seconds=5), b.recv_json())
        assert recvd == obj
    self.loop.run_sync(test)