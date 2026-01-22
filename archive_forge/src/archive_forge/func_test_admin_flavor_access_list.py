from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_flavor_access_list(self):
    self.assertRaises(exceptions.CommandFailed, self.nova, 'flavor-access-list')
    self.assertRaises(exceptions.CommandFailed, self.nova, 'flavor-access-list', params='--flavor m1.tiny')