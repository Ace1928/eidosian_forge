from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_unset_flow_classifier(self):
    target = self.resource['id']
    fc1 = 'flow_classifier1'
    self.network.find_sfc_port_chain = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'flow_classifiers': [self.pc_fc]})
    self.network.find_sfc_flow_classifier.side_effect = lambda name_or_id, ignore_missing=False: {'id': name_or_id}
    arglist = [target, '--flow-classifier', fc1]
    verifylist = [(self.res, target), ('flow_classifiers', [fc1])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    expect = {'flow_classifiers': [self.pc_fc]}
    self.mocked.assert_called_once_with(target, **expect)
    self.assertIsNone(result)