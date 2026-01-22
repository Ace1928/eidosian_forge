import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data(*set(['2.45', api_versions.MAX_VERSION]))
def test_create_update_show_access_rule_with_metadata(self, microversion):
    self.skip_if_microversion_not_supported(microversion)
    md1 = {'key1': 'value1', 'key2': 'value2'}
    md2 = {'key3': 'value3', 'key2': 'value4'}
    access = self._test_create_list_access_rule_for_share(metadata=md1, microversion=microversion)
    get_access = self.user_client.access_show(access['id'], microversion=microversion)
    self.assertEqual(access['id'], get_access['id'])
    self.assertEqual(md1, ast.literal_eval(get_access['metadata']))
    self.user_client.access_set_metadata(access['id'], metadata=md2, microversion=microversion)
    get_access = self.user_client.access_show(access['id'], microversion=microversion)
    self.assertEqual({'key1': 'value1', 'key2': 'value4', 'key3': 'value3'}, ast.literal_eval(get_access['metadata']))
    self.assertEqual(access['id'], get_access['id'])