import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_get_and_delete_share_server(self):
    self.share, share_network = self._create_share_and_share_network()
    share_server_id = self.client.get_share(self.share['id'])['share_server_id']
    server = self.client.get_share_server(share_server_id)
    expected_keys = ('id', 'host', 'status', 'created_at', 'updated_at', 'share_network_id', 'share_network_name', 'project_id')
    if utils.is_microversion_supported('2.49'):
        expected_keys += ('identifier', 'is_auto_deletable')
    for key in expected_keys:
        self.assertIn(key, server)
    self._delete_share_and_share_server(self.share['id'], share_server_id)
    self.client.delete_share_network(share_network['id'])