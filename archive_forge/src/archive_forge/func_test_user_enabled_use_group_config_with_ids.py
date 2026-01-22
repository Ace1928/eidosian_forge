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
def test_user_enabled_use_group_config_with_ids(self):
    group_name = 'enabled_users'
    driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
    group_dn = 'cn=%s,%s' % (group_name, driver.group.tree_dn)
    self.config_fixture.config(group='ldap', user_enabled_emulation_use_group_config=True, user_enabled_emulation_dn=group_dn, group_name_attribute='cn', group_member_attribute='memberUid', group_members_are_ids=True, group_objectclass='posixGroup')
    self.ldapdb.clear()
    self.load_backends()
    user1 = unit.new_user_ref(enabled=True, domain_id=CONF.identity.default_domain_id)
    user_ref = PROVIDERS.identity_api.create_user(user1)
    self.assertIs(True, user_ref['enabled'])
    user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
    self.assertIs(True, user_ref['enabled'])
    group_ref = PROVIDERS.identity_api.get_group_by_name(group_name, CONF.identity.default_domain_id)
    PROVIDERS.identity_api.check_user_in_group(user_ref['id'], group_ref['id'])