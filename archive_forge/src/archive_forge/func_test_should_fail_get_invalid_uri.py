from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_fail_get_invalid_uri(self):
    self.assertRaises(ValueError, self.manager.get, self.secret_acl_ref)
    self.assertRaises(ValueError, self.manager.get, self.endpoint + '/containers/consumers')