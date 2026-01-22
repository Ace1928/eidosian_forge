import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_queue_delete_functional(self):
    queue = self.client.queue('nonono')
    queue._get_transport = mock.Mock(return_value=self.transport)
    messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 3!'}]
    queue.post(messages)
    queue.delete()
    self.assertEqual(0, len(list(queue.messages(echo=True))))