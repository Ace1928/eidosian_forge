from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_group_types as osc_share_group_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_type_create_required_args(self):
    """Verifies required arguments."""
    arglist = [self.share_group_type.name, self.share_types[0].name, self.share_types[1].name]
    verifylist = [('name', self.share_group_type.name), ('share_types', [self.share_types[0].name, self.share_types[1].name])]
    with mock.patch('manilaclient.common.apiclient.utils.find_resource', side_effect=[self.share_types[0], self.share_types[1]]):
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sgt_mock.create.assert_called_with(group_specs={}, is_public=True, name=self.share_group_type.name, share_types=[self.share_types[0].name, self.share_types[1].name])
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)