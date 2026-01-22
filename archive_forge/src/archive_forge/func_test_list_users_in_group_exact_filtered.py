import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_users_in_group_exact_filtered(self):
    hints = driver_hints.Hints()
    user_list, group = self._list_users_in_group_data()
    hints.add_filter('name', 'Arthur Rimbaud', comparator='equals')
    users = PROVIDERS.identity_api.list_users_in_group(group['id'], hints=hints)
    self.assertEqual(1, len(users))
    self.assertEqual(user_list[3]['id'], users[0]['id'])
    self._delete_test_data('user', user_list)
    self._delete_entity('group')(group['id'])