from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_snapshots_detail(self):
    values = (oscutils.get_dict_properties(s._info, COLUMNS_DETAIL) for s in self.snapshots_list)
    arglist = ['--detail']
    verifylist = [('detail', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.list.assert_called_with(search_opts={'offset': None, 'limit': None, 'all_tenants': False, 'name': None, 'status': None, 'share_id': None, 'usage': None, 'metadata': {}, 'name~': None, 'description~': None, 'description': None})
    self.assertEqual(COLUMNS_DETAIL, columns)
    self.assertEqual(list(values), list(data))