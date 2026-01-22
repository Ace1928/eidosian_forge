from unittest import mock
import iso8601
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_event
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_server_event_show(self):
    arglist = [self.fake_server.name, self.fake_event.request_id]
    verifylist = [('server', self.fake_server.name), ('request_id', self.fake_event.request_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_server.assert_called_with(self.fake_server.name, ignore_missing=False)
    self.compute_sdk_client.get_server_action.assert_called_with(self.fake_event.request_id, self.fake_server.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)