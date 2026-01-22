import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_set_address(self):
    new_portgroup_address = '00:22:44:66:88:00'
    arglist = [baremetal_fakes.baremetal_portgroup_uuid, '--address', new_portgroup_address]
    verifylist = [('portgroup', baremetal_fakes.baremetal_portgroup_uuid), ('address', new_portgroup_address)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with(baremetal_fakes.baremetal_portgroup_uuid, [{'path': '/address', 'value': new_portgroup_address, 'op': 'add'}])