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
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'simple_bind_s')
def test_user_api_get_connection_no_user_password(self, mocked_method):
    """Bind anonymously when the user and password are blank."""
    self.config_fixture.config(group='ldap', user=None, password=None)
    user_api = identity.backends.ldap.UserApi(CONF)
    user_api.get_connection(user=None, password=None)
    self.assertTrue(mocked_method.called)