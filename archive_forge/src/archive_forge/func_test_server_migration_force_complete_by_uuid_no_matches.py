from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_server_migration_force_complete_by_uuid_no_matches(self):
    self._set_mock_microversion('2.59')
    self.compute_sdk_client.server_migrations.return_value = iter([])
    arglist = [self.server.id, '69f95745-bfe3-4302-90f7-5b0022cba1ce']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('In-progress live migration 69f95745-bfe3-4302-90f7-5b0022cba1ce', str(ex))