from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_pools
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_pools_share_type(self):
    arglist = ['--share-type', self.share_type.id]
    verifylist = [('share_type', self.share_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.pools_mock.list.assert_called_with(detailed=False, search_opts={'host': None, 'backend': None, 'pool': None, 'share_type': self.share_type.id})
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))