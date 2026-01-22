import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_message_list(self):
    returned = {'links': [{'rel': 'next', 'href': '/v1/queues/fizbit/messages?marker=6244-244224-783'}], 'messages': [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]}
    with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
        resp = response.Response(None, json.dumps(returned))
        send_method.return_value = resp
        msgs = self.queue.messages(limit=1)
        self.assertIsInstance(msgs, iterator._Iterator)