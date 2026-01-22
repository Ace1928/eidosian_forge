from unittest import mock
import iso8601
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_event
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_server_event_list_with_changes_before(self):
    self._set_mock_microversion('2.66')
    arglist = ['--changes-before', '2016-03-04T06:27:59Z', self.fake_server.name]
    verifylist = [('server', self.fake_server.name), ('changes_before', '2016-03-04T06:27:59Z')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_server.assert_called_with(self.fake_server.name, ignore_missing=False)
    self.compute_sdk_client.server_actions.assert_called_with(self.fake_server.id, changes_before='2016-03-04T06:27:59Z')
    self.assertEqual(self.columns, columns)
    self.assertEqual(tuple(self.data), tuple(data))