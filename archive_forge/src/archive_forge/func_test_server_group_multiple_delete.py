from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_server_group_multiple_delete(self):
    arglist = ['affinity_group', 'anti_affinity_group']
    verifylist = [('server_group', ['affinity_group', 'anti_affinity_group'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_server_group.assert_any_call('affinity_group', ignore_missing=False)
    self.compute_sdk_client.find_server_group.assert_any_call('anti_affinity_group', ignore_missing=False)
    self.compute_sdk_client.delete_server_group.assert_called_with(self.fake_server_group.id)
    self.assertEqual(2, self.compute_sdk_client.find_server_group.call_count)
    self.assertEqual(2, self.compute_sdk_client.delete_server_group.call_count)
    self.assertIsNone(result)