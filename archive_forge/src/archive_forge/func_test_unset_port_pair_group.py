from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_unset_port_pair_group(self):
    target = self.resource['id']
    ppg1 = 'port_pair_group1'
    self.network.find_sfc_port_chain = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'port_pair_groups': [self.pc_ppg]})
    self.network.find_sfc_port_pair_group.side_effect = lambda name_or_id, ignore_missing=False: {'id': name_or_id}
    arglist = [target, '--port-pair-group', ppg1]
    verifylist = [(self.res, target), ('port_pair_groups', [ppg1])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    expect = {'port_pair_groups': [self.pc_ppg]}
    self.mocked.assert_called_once_with(target, **expect)
    self.assertIsNone(result)