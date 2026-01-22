from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_list(self):
    self.nova('list')
    self.nova('list', params='--all-tenants 1')
    self.nova('list', params='--all-tenants 0')
    self.assertRaises(exceptions.CommandFailed, self.nova, 'list', params='--all-tenants bad')