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
def test_user_name_in_dn(self, mock_ldap_get):
    driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
    driver.user.id_attr = 'SAMAccountName'
    driver.user.attribute_mapping['name'] = 'cn'
    mock_ldap_get.return_value = ('cn=Foo Bar,dc=example,dc=com', {'sn': [uuid.uuid4().hex], 'cn': ['Foo Bar'], 'SAMAccountName': ['crap']})
    user_ref = PROVIDERS.identity_api.get_user('crap')
    self.assertEqual('crap', user_ref['id'])
    self.assertEqual('Foo Bar', user_ref['name'])