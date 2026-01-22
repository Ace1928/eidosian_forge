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
@mock.patch.object(common_ldap.BaseLdap, '_ldap_get')
def test_user_mixed_case_attribute(self, mock_ldap_get):
    mock_ldap_get.return_value = ('cn=junk,dc=example,dc=com', {'sN': [uuid.uuid4().hex], 'MaIl': [uuid.uuid4().hex], 'cn': ['junk']})
    user = PROVIDERS.identity_api.get_user('junk')
    self.assertEqual(mock_ldap_get.return_value[1]['sN'][0], user['name'])
    self.assertEqual(mock_ldap_get.return_value[1]['MaIl'][0], user['email'])