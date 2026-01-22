from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_unset_all_flow_classifier(self):
    client = self.app.client_manager.network
    target = self.resource['id']
    mock_port_chain_update = client.update_sfc_port_chain
    arglist = [target, '--all-flow-classifier']
    verifylist = [(self.res, target), ('all_flow_classifier', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    expect = {'flow_classifiers': []}
    mock_port_chain_update.assert_called_once_with(target, **expect)
    self.assertIsNone(result)