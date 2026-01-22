import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_port_create_uuid(self):
    port_uuid = 'da6c8d2e-fbcd-457a-b2a7-cc5c775933af'
    arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid, '--uuid', port_uuid]
    verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address), ('uuid', port_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'uuid': port_uuid}
    self.baremetal_mock.port.create.assert_called_once_with(**args)