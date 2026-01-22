import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_subscription_create(self):
    self.assertEqual('http://trigger.me', self.subscription_1.subscriber)
    self.assertEqual(3600, self.subscription_1.ttl)
    self.assertEqual('http://trigger.he', self.subscription_2.subscriber)
    self.assertEqual(7200, self.subscription_2.ttl)