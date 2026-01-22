from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_role_set(self):
    role_name = self._create_dummy_role()
    new_role_name = data_utils.rand_name('NewTestRole')
    raw_output = self.openstack('role set --name %s %s' % (new_role_name, role_name))
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('role show %s' % new_role_name)
    role = self.parse_show_as_object(raw_output)
    self.assertEqual(new_role_name, role['name'])