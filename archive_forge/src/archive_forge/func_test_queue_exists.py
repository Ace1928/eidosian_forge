import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_queue_exists(self):
    with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
        resp = response.Response(None, None)
        send_method.return_value = resp
        self.queue.exists()