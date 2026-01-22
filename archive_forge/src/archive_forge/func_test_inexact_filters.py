import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_inexact_filters(self):
    user_list = self._create_test_data('user', 20)
    user = user_list[5]
    user['name'] = 'The'
    PROVIDERS.identity_api.update_user(user['id'], user)
    user = user_list[6]
    user['name'] = 'The Ministry'
    PROVIDERS.identity_api.update_user(user['id'], user)
    user = user_list[7]
    user['name'] = 'The Ministry of'
    PROVIDERS.identity_api.update_user(user['id'], user)
    user = user_list[8]
    user['name'] = 'The Ministry of Silly'
    PROVIDERS.identity_api.update_user(user['id'], user)
    user = user_list[9]
    user['name'] = 'The Ministry of Silly Walks'
    PROVIDERS.identity_api.update_user(user['id'], user)
    user = user_list[10]
    user['name'] = 'the ministry of silly walks OF'
    PROVIDERS.identity_api.update_user(user['id'], user)
    self._set_policy({'identity:list_users': []})
    url_by_name = '/users?name__contains=Ministry'
    r = self.get(url_by_name, auth=self.auth)
    self.assertEqual(4, len(r.result.get('users')))
    self._match_with_list(r.result.get('users'), user_list, list_start=6, list_end=10)
    url_by_name = '/users?name__icontains=miNIstry'
    r = self.get(url_by_name, auth=self.auth)
    self.assertEqual(5, len(r.result.get('users')))
    self._match_with_list(r.result.get('users'), user_list, list_start=6, list_end=11)
    url_by_name = '/users?name__startswith=The'
    r = self.get(url_by_name, auth=self.auth)
    self.assertEqual(5, len(r.result.get('users')))
    self._match_with_list(r.result.get('users'), user_list, list_start=5, list_end=10)
    url_by_name = '/users?name__istartswith=the'
    r = self.get(url_by_name, auth=self.auth)
    self.assertEqual(6, len(r.result.get('users')))
    self._match_with_list(r.result.get('users'), user_list, list_start=5, list_end=11)
    url_by_name = '/users?name__endswith=of'
    r = self.get(url_by_name, auth=self.auth)
    self.assertEqual(1, len(r.result.get('users')))
    self.assertEqual(user_list[7]['id'], r.result.get('users')[0]['id'])
    url_by_name = '/users?name__iendswith=OF'
    r = self.get(url_by_name, auth=self.auth)
    self.assertEqual(2, len(r.result.get('users')))
    self.assertEqual(user_list[7]['id'], r.result.get('users')[0]['id'])
    self.assertEqual(user_list[10]['id'], r.result.get('users')[1]['id'])
    self._delete_test_data('user', user_list)