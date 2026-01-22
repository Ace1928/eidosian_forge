from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_set_service_graph(self):
    client = self.app.client_manager.network
    mock_service_graph_update = client.update_sfc_service_graph
    arglist = [self._service_graph_name, '--name', 'name_updated', '--description', 'desc_updated']
    verifylist = [('service_graph', self._service_graph_name), ('name', 'name_updated'), ('description', 'desc_updated')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'name_updated', 'description': 'desc_updated'}
    mock_service_graph_update.assert_called_once_with(self._service_graph_name, **attrs)
    self.assertIsNone(result)