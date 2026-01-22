import uuid
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def test_quota_network_set_with_no_force(self):
    if not self.haz_network:
        self.skipTest('No Network service present')
    if not self.is_extension_enabled('quota-check-limit'):
        self.skipTest('No "quota-check-limit" extension present')
    cmd_output = self.openstack('quota list --network', parse_output=True)
    self.addCleanup(self._restore_quota_limit, 'network', cmd_output[0]['Networks'], self.PROJECT_NAME)
    self.openstack('quota set --networks 40 ' + self.PROJECT_NAME)
    cmd_output = self.openstack('quota list --network', parse_output=True)
    self.assertIsNotNone(cmd_output)
    self.assertEqual(40, cmd_output[0]['Networks'])
    for _ in range(2):
        self.openstack('network create --project %s %s' % (self.PROJECT_NAME, uuid.uuid4().hex))
    self.assertRaises(exceptions.CommandFailed, self.openstack, 'quota set --networks 1 --no-force ' + self.PROJECT_NAME)