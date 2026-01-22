import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('1.0', '2.0', '2.6', '2.7', '2.21', '2.33')
def test_create_delete_user_access_rule(self, microversion):
    self._create_delete_access_rule(self.share_id, 'user', CONF.username_for_user_rules, microversion)