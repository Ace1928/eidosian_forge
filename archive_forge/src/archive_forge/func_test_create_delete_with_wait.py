import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_create_delete_with_wait(self):
    name = data_utils.rand_name('share-with-wait-%s')
    description = data_utils.rand_name('we-wait-until-share-is-ready')
    share_1, share_2 = (self.create_share(self.protocol, name=name % num, description=description, use_wait_option=True, client=self.user_client) for num in range(0, 2))
    self.assertEqual('available', share_1['status'])
    self.assertEqual('available', share_2['status'])
    self.delete_share([share_1['id'], share_2['id']], wait=True, client=self.user_client)
    for share in (share_1, share_2):
        self.assertRaises(exceptions.NotFound, self.user_client.get_share, share['id'])