from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_set_flow_classifier(self):
    client = self.app.client_manager.network
    mock_flow_classifier_update = client.update_sfc_flow_classifier
    arglist = [self._flow_classifier_name, '--name', 'name_updated', '--description', 'desc_updated']
    verifylist = [('flow_classifier', self._flow_classifier_name), ('name', 'name_updated'), ('description', 'desc_updated')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'name_updated', 'description': 'desc_updated'}
    mock_flow_classifier_update.assert_called_once_with(self._flow_classifier_name, **attrs)
    self.assertIsNone(result)