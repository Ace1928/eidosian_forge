from unittest import mock
import iso8601
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_event
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_server_event_list_with_limit(self):
    self._set_mock_microversion('2.58')
    arglist = ['--limit', '1', self.fake_server.name]
    verifylist = [('limit', 1), ('server', self.fake_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.compute_sdk_client.server_actions.assert_called_with(self.fake_server.id, limit=1, paginated=False)