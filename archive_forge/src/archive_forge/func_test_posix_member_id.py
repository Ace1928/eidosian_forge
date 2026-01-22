import copy
from unittest import mock
import uuid
import fixtures
import http.client
import ldap
from oslo_log import versionutils
import pkg_resources
from testtools import matchers
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.identity.backends import ldap as ldap_identity
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends import sql as sql_identity
from keystone.identity.mapping_backends import mapping as map
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.resource import test_backends as resource_tests
def test_posix_member_id(self):
    domain = self._get_domain_fixture()
    new_group = unit.new_group_ref(domain_id=domain['id'])
    new_group = PROVIDERS.identity_api.create_group(new_group)
    user_refs = PROVIDERS.identity_api.list_users_in_group(new_group['id'])
    self.assertEqual([], user_refs)
    new_user = unit.new_user_ref(domain_id=domain['id'])
    new_user = PROVIDERS.identity_api.create_user(new_user)
    group_api = PROVIDERS.identity_api.driver.group
    group_ref = group_api.get(new_group['id'])
    mod = (ldap.MOD_ADD, group_api.member_attribute, new_user['id'])
    conn = group_api.get_connection()
    conn.modify_s(group_ref['dn'], [mod])
    user_refs = PROVIDERS.identity_api.list_users_in_group(new_group['id'])
    self.assertIn(new_user['id'], (x['id'] for x in user_refs))
    group_refs = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
    self.assertIn(new_group['id'], (x['id'] for x in group_refs))