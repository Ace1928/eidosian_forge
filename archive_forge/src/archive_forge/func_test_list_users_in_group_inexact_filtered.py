import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_users_in_group_inexact_filtered(self):
    user_list, group = self._list_users_in_group_data()
    hints = driver_hints.Hints()
    hints.add_filter('name', 'Arthur', comparator='contains')
    users = PROVIDERS.identity_api.list_users_in_group(group['id'], hints=hints)
    self.assertThat(len(users), matchers.Equals(2))
    self.assertIn(user_list[1]['id'], [users[0]['id'], users[1]['id']])
    self.assertIn(user_list[3]['id'], [users[0]['id'], users[1]['id']])
    hints = driver_hints.Hints()
    hints.add_filter('name', 'Arthur', comparator='startswith')
    users = PROVIDERS.identity_api.list_users_in_group(group['id'], hints=hints)
    self.assertThat(len(users), matchers.Equals(2))
    self.assertIn(user_list[1]['id'], [users[0]['id'], users[1]['id']])
    self.assertIn(user_list[3]['id'], [users[0]['id'], users[1]['id']])
    hints = driver_hints.Hints()
    hints.add_filter('name', 'Doyle', comparator='endswith')
    users = PROVIDERS.identity_api.list_users_in_group(group['id'], hints=hints)
    self.assertThat(len(users), matchers.Equals(1))
    self.assertEqual(user_list[1]['id'], users[0]['id'])
    self._delete_test_data('user', user_list)
    self._delete_entity('group')(group['id'])