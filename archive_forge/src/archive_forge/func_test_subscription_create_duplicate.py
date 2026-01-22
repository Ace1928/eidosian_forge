import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_subscription_create_duplicate(self):
    subscription_data_2 = {'subscriber': 'http://trigger.he', 'ttl': 7200, 'options': {'check everything': True}}
    new_subscription = self.client.subscription('beijing', **subscription_data_2)
    self.assertEqual(new_subscription.id, self.subscription_2.id)