import ddt
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_delete_no_node(self):
    """Test for baremetal node delete without node specified."""
    command = 'baremetal node delete'
    ex_text = ''
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)