import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@testtools.skipUnless(CONF.share_network, 'Usage of Share networks is disabled')
def test_list_shares_by_share_network(self):
    share_network_id = self.user_client.get_share_network(CONF.share_network)['id']
    self._list_shares({'share_network': share_network_id})