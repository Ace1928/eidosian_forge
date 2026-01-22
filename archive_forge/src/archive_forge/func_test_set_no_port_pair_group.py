from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_set_no_port_pair_group(self):
    target = self.resource['id']
    ppg1 = 'port_pair_group1'
    arglist = [target, '--no-port-pair-group', '--port-pair-group', ppg1]
    verifylist = [(self.res, target), ('no_port_pair_group', True), ('port_pair_groups', [ppg1])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    expect = {'port_pair_groups': [ppg1]}
    self.mocked.assert_called_once_with(target, **expect)
    self.assertIsNone(result)