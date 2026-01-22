import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_subscription_get(self):
    kwargs = {'id': self.subscription_1.id}
    subscription_get = self.client.subscription(self.queue_name, **kwargs)
    self.assertEqual('http://trigger.me', subscription_get.subscriber)
    self.assertEqual(3600, subscription_get.ttl)