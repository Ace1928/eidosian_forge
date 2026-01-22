from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_list_port_pair_groups(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns = self.cmd.take_action(parsed_args)[0]
    ppgs = self.network.sfc_port_pair_groups()
    ppg = ppgs[0]
    data = [ppg['id'], ppg['name'], ppg['port_pairs'], ppg['port_pair_group_parameters'], ppg['tap_enabled']]
    self.assertEqual(list(self.columns), columns)
    self.assertEqual(self.data, data)