from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_flavor_list(self):
    self.assertIn('Memory_MiB', self.nova('flavor-list'))