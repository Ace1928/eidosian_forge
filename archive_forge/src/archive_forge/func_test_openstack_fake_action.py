import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_fake_action(self):
    self.assertRaises(exceptions.CommandFailed, self.openstack, 'this-does-not-exist')