import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_subscription_create_duplicate_throws_conflicterror(self):
    subscription_data = {'subscriber': 'http://trigger.me', 'ttl': 3600}
    with mock.patch.object(self.transport.client, 'request', autospec=True) as request_method:

        class FakeRawResponse(object):

            def __init__(self):
                self.text = ''
                self.headers = {}
                self.status_code = 409
        request_method.return_value = FakeRawResponse()
        self.assertRaises(errors.ConflictError, self.client.subscription, 'beijing', **subscription_data)