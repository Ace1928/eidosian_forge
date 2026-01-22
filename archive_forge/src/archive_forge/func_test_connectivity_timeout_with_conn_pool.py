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
def test_connectivity_timeout_with_conn_pool(self, mock_ldap_bind):
    url = 'ldap://localhost'
    conn_timeout = 1
    self.config_fixture.config(group='ldap', url=url, pool_connection_timeout=conn_timeout, use_pool=True, pool_retry_max=1)
    base_ldap = common_ldap.BaseLdap(CONF)
    ldap_connection = base_ldap.get_connection()
    self.assertIsInstance(ldap_connection.conn, common_ldap.PooledLDAPHandler)
    self.assertEqual(conn_timeout, ldap.get_option(ldap.OPT_NETWORK_TIMEOUT))
    self.assertEqual(url, ldap_connection.conn.conn_pool.uri)