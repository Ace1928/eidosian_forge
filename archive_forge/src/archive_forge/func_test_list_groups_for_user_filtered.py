import subprocess
import ldap.modlist
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.identity.backends import ldap as identity_ldap
from keystone.tests import unit
from keystone.tests.unit import test_backend_ldap
def test_list_groups_for_user_filtered(self):
    domain = self._get_domain_fixture()
    test_groups = []
    test_users = []
    GROUP_COUNT = 3
    USER_COUNT = 2
    positive_user = unit.create_user(PROVIDERS.identity_api, domain['id'])
    negative_user = unit.create_user(PROVIDERS.identity_api, domain['id'])
    for x in range(0, USER_COUNT):
        group_refs = PROVIDERS.identity_api.list_groups_for_user(test_users[x]['id'])
        self.assertEqual(0, len(group_refs))
    for x in range(0, GROUP_COUNT):
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        test_groups.append(new_group)
        group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
        self.assertEqual(x, len(group_refs))
        PROVIDERS.identity_api.add_user_to_group(positive_user['id'], new_group['id'])
        group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
        self.assertEqual(x + 1, len(group_refs))
        group_refs = PROVIDERS.identity_api.list_groups_for_user(negative_user['id'])
        self.assertEqual(0, len(group_refs))
    driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
    driver.group.ldap_filter = '(dn=xx)'
    group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
    self.assertEqual(0, len(group_refs))
    group_refs = PROVIDERS.identity_api.list_groups_for_user(negative_user['id'])
    self.assertEqual(0, len(group_refs))
    driver.group.ldap_filter = '(objectclass=*)'
    group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
    self.assertEqual(GROUP_COUNT, len(group_refs))
    group_refs = PROVIDERS.identity_api.list_groups_for_user(negative_user['id'])
    self.assertEqual(0, len(group_refs))