import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_create_mode_properties(self):
    arglist = ['--node', baremetal_fakes.baremetal_uuid, '--mode', baremetal_fakes.baremetal_portgroup_mode, '--property', 'key1=value11', '--property', 'key2=value22']
    verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('mode', baremetal_fakes.baremetal_portgroup_mode), ('properties', ['key1=value11', 'key2=value22'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'mode': baremetal_fakes.baremetal_portgroup_mode, 'properties': baremetal_fakes.baremetal_portgroup_properties}
    self.baremetal_mock.portgroup.create.assert_called_once_with(**args)