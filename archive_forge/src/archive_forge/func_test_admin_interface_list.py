from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
@decorators.skip_because(bug='1157349')
def test_admin_interface_list(self):
    self.nova('interface-list')