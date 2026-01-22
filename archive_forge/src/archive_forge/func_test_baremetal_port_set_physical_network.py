import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_set_physical_network(self):
    new_physical_network = 'physnet2'
    arglist = [baremetal_fakes.baremetal_port_uuid, '--physical-network', new_physical_network]
    verifylist = [('port', baremetal_fakes.baremetal_port_uuid), ('physical_network', new_physical_network)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.port.update.assert_called_once_with(baremetal_fakes.baremetal_port_uuid, [{'path': '/physical_network', 'value': new_physical_network, 'op': 'add'}])