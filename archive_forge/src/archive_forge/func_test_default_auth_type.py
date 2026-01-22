from tempest.lib import exceptions as tempest_exc
from openstackclient.tests.functional import base
def test_default_auth_type(self):
    cmd_output = self.openstack('configuration show', cloud='', parse_output=True)
    self.assertIsNotNone(cmd_output)
    self.assertIn('auth_type', cmd_output.keys())
    self.assertEqual('password', cmd_output['auth_type'])