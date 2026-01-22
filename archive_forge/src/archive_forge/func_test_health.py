import json
from unittest import mock
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_health(self):
    health = self.client.health()
    self.assertIsInstance(health, dict)