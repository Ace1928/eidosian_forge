import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_groups_for_user(self):
    domain = self._get_domain_fixture()
    test_groups = []
    test_users = []
    GROUP_COUNT = 3
    USER_COUNT = 2
    for x in range(0, USER_COUNT):
        new_user = unit.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        test_users.append(new_user)
    positive_user = test_users[0]
    negative_user = test_users[1]
    for x in range(0, USER_COUNT):
        group_refs = PROVIDERS.identity_api.list_groups_for_user(test_users[x]['id'])
        self.assertEqual(0, len(group_refs))
    for x in range(0, GROUP_COUNT):
        before_count = x
        after_count = x + 1
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        test_groups.append(new_group)
        group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
        self.assertEqual(before_count, len(group_refs))
        PROVIDERS.identity_api.add_user_to_group(positive_user['id'], new_group['id'])
        group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
        self.assertEqual(after_count, len(group_refs))
        group_refs = PROVIDERS.identity_api.list_groups_for_user(negative_user['id'])
        self.assertEqual(0, len(group_refs))