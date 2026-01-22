from oslo_utils import timeutils
from barbicanclient.tests import test_client
from barbicanclient.v1 import cas
def test_should_fail_get_invalid_ca(self):
    self.assertRaises(ValueError, self.manager.get, **{'ca_ref': '12345'})