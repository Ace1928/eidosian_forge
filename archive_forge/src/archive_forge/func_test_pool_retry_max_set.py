from unittest import mock
import fixtures
import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap
def test_pool_retry_max_set(self):
    ldappool_cm = self.conn_pools[CONF.ldap.url]
    self.assertEqual(CONF.ldap.pool_retry_max, ldappool_cm.retry_max)