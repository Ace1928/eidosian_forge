from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_create(self):
    self.start_server()
    self.assertEqual(201, self._create().status_code)
    self.set_policy_rules({'add_image': '!'})
    self.assertEqual(403, self._create().status_code)