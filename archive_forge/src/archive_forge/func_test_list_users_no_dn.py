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
def test_list_users_no_dn(self):
    users = PROVIDERS.identity_api.list_users()
    self.assertEqual(len(default_fixtures.USERS), len(users))
    user_ids = set((user['id'] for user in users))
    expected_user_ids = set((getattr(self, 'user_%s' % user['name'])['id'] for user in default_fixtures.USERS))
    for user_ref in users:
        self.assertNotIn('dn', user_ref)
    self.assertEqual(expected_user_ids, user_ids)