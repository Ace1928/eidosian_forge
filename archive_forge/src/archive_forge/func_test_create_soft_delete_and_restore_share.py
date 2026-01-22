import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_create_soft_delete_and_restore_share(self):
    self.skip_if_microversion_not_supported('2.69')
    microversion = '2.69'
    description = data_utils.rand_name('we-wait-until-share-is-ready')
    share = self.create_share(self.protocol, name='share_name', description=description, use_wait_option=True, client=self.user_client)
    self.assertEqual('available', share['status'])
    self.soft_delete_share([share['id']], client=self.user_client, microversion=microversion)
    self.user_client.wait_for_share_soft_deletion(share['id'])
    result = self.user_client.list_shares(is_soft_deleted=True, microversion=microversion)
    share_ids = [sh['ID'] for sh in result]
    self.assertIn(share['id'], share_ids)
    self.restore_share([share['id']], client=self.user_client, microversion=microversion)
    self.user_client.wait_for_share_restore(share['id'])
    result1 = self.user_client.list_shares(is_soft_deleted=True, microversion=microversion)
    share_ids1 = [sh['ID'] for sh in result1]
    self.assertNotIn(share['id'], share_ids1)