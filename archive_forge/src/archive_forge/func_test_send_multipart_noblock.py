import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_send_multipart_noblock(self):

    async def test():
        s = self.socket(zmq.PUSH)
        with pytest.raises(zmq.Again):
            await s.send_multipart([b'not going anywhere'], flags=zmq.NOBLOCK)
    self.loop.run_sync(test)