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
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'connect')
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'search_s')
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'simple_bind_s')
def test_filter_ldap_result_by_attr(self, mock_simple_bind_s, mock_search_s, mock_connect):
    mock_search_s.return_value = [('sn=junk1,dc=example,dc=com', {'cn': [uuid.uuid4().hex], 'email': [uuid.uuid4().hex], 'sn': ['junk1']}), ('', {'cn': [uuid.uuid4().hex], 'email': [uuid.uuid4().hex]}), ('sn=,dc=example,dc=com', {'cn': [uuid.uuid4().hex], 'email': [uuid.uuid4().hex], 'sn': ['']}), ('sn=   ,dc=example,dc=com', {'cn': [uuid.uuid4().hex], 'email': [uuid.uuid4().hex], 'sn': ['   ']})]
    user_api = identity.backends.ldap.UserApi(CONF)
    user_refs = user_api.get_all()
    self.assertEqual(1, len(user_refs))
    self.assertEqual('junk1', user_refs[0]['name'])
    self.assertEqual('sn=junk1,dc=example,dc=com', user_refs[0]['dn'])