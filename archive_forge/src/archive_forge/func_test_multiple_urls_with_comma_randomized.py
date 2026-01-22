import os
import tempfile
from unittest import mock
import uuid
import fixtures
import ldap.dn
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception as ks_exception
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import fakeldap
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'simple_bind_s')
def test_multiple_urls_with_comma_randomized(self, mock_ldap_bind):
    urls = 'ldap://localhost1,ldap://localhost2,ldap://localhost3,ldap://localhost4,ldap://localhost5,ldap://localhost6,ldap://localhost7,ldap://localhost8,ldap://localhost9,ldap://localhost0'
    self.config_fixture.config(group='ldap', url=urls, randomize_urls=True)
    base_ldap = common_ldap.BaseLdap(CONF)
    ldap_connection = base_ldap.get_connection()
    self.assertEqual(len(urls.split(',')), 10)
    self.assertEqual(len(urls.split(',')), len(ldap_connection.conn.conn_pool.uri.split(',')))
    self.assertNotEqual(urls.split(','), ldap_connection.conn.conn_pool.uri.split(','))
    self.assertEqual(set(urls.split(',')), set(ldap_connection.conn.conn_pool.uri.split(',')))