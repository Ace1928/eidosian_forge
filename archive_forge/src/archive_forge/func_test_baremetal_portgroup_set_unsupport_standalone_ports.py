import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_set_unsupport_standalone_ports(self):
    arglist = [baremetal_fakes.baremetal_portgroup_uuid, '--unsupport-standalone-ports']
    verifylist = [('portgroup', baremetal_fakes.baremetal_portgroup_uuid), ('unsupport_standalone_ports', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with(baremetal_fakes.baremetal_portgroup_uuid, [{'path': '/standalone_ports_supported', 'value': 'False', 'op': 'add'}])