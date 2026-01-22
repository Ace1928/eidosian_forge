import ddt
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
@ddt.data(('--property', '', 'error: the following arguments are required: <node>'), ('--property', 'prop', "Reason: can't remove a non-existent object"))
@ddt.unpack
def test_unset_property(self, argument, value, ex_text):
    """Negative test for baremetal node unset command options."""
    base_cmd = 'baremetal node unset'
    command = self.construct_cmd(base_cmd, argument, value, self.node['uuid'])
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)