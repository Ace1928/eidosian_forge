import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_portgroup_set_multiple_properties(self):
    arglist = ['portgroup', '--property', 'key3=val3', '--property', 'key4=val4']
    verifylist = [('portgroup', 'portgroup'), ('properties', ['key3=val3', 'key4=val4'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/properties/key3', 'value': 'val3', 'op': 'add'}, {'path': '/properties/key4', 'value': 'val4', 'op': 'add'}])