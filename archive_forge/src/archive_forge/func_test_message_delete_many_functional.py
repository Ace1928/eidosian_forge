import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_message_delete_many_functional(self):
    queue = self.client.queue('test_queue')
    self.addCleanup(queue.delete)
    queue._get_transport = mock.Mock(return_value=self.transport)
    messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}]
    res = queue.post(messages)['resources']
    msgs_id = [ref.split('/')[-1] for ref in res]
    queue.delete_messages(*msgs_id)
    messages = queue.messages()
    self.assertEqual(0, len(list(messages)))