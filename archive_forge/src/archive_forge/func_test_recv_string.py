import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_recv_string(self):

    async def test():
        a, b = self.create_bound_pair(zmq.PUSH, zmq.PULL)
        f = b.recv_string()
        assert not f.done()
        msg = 'πøøπ'
        await a.send_string(msg)
        recvd = await f
        assert f.done()
        assert f.result() == msg
        assert recvd == msg
    self.loop.run_sync(test)