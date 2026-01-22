import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_queue_stats_functional(self):
    messages = [{'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}]
    queue = self.client.queue('nonono')
    self.addCleanup(queue.delete)
    queue._get_transport = mock.Mock(return_value=self.transport)
    queue.post(messages)
    stats = queue.stats
    self.assertEqual(3, stats['messages']['free'])