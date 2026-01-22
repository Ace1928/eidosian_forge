import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_create_smartnic(self):
    arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid, '--is-smartnic']
    verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address), ('is_smartnic', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'is_smartnic': True}
    self.baremetal_mock.port.create.assert_called_once_with(**args)