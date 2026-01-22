import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data(*set(['1.0', '2.0', '2.6', '2.7', '2.21', '2.33', '2.44', '2.45', api_versions.MAX_VERSION]))
def test_create_list_access_rule_for_share(self, microversion):
    self.skip_if_microversion_not_supported(microversion)
    access = self._test_create_list_access_rule_for_share(microversion=microversion)
    access_list = self.user_client.list_access(self.share['id'], microversion=microversion)
    self.assertTrue(any([item for item in access_list if access['id'] == item['id']]))
    self.assertTrue(any((a['access_type'] is not None for a in access_list)))
    self.assertTrue(any((a['access_to'] is not None for a in access_list)))
    self.assertTrue(any((a['access_level'] is not None for a in access_list)))
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.33'):
        self.assertTrue(all((all((key in access for key in ('access_key', 'created_at', 'updated_at'))) for access in access_list)))
    elif api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.21'):
        self.assertTrue(all(('access_key' in a for a in access_list)))
    else:
        self.assertTrue(all(('access_key' not in a for a in access_list)))