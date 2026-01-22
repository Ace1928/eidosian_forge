from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_types as osc_share_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_type_create_bool_args(self):
    arglist = [self.new_share_type.name, 'True', '--snapshot-support', 'true']
    verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True'), ('snapshot_support', 'true')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.shares_mock.create.assert_called_with(extra_specs={'snapshot_support': 'True'}, is_public=True, name=self.new_share_type.name, spec_driver_handles_share_servers=True)
    self.assertCountEqual(COLUMNS, columns)
    self.assertCountEqual(self.data, data)