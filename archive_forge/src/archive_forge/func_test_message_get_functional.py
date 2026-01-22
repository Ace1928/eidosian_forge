import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_message_get_functional(self):
    queue = self.client.queue('test_queue')
    self.addCleanup(queue.delete)
    queue._get_transport = mock.Mock(return_value=self.transport)
    messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 3!'}]
    res = queue.post(messages)['resources']
    msg_id = res[0].split('/')[-1]
    msg = queue.message(msg_id)
    self.assertIsInstance(msg, message.Message)
    self.assertEqual(res[0], msg.href)