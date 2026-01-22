from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_fail_create_invalid_uri(self):
    self.assertRaises(ValueError, self.manager.create, self.endpoint + '/orders')
    self.assertRaises(ValueError, self.manager.create, None)