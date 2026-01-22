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
def test_handler_with_use_pool_enabled(self):
    user_ref = PROVIDERS.identity_api.get_user(self.user_foo['id'])
    self.user_foo.pop('password')
    self.assertDictEqual(self.user_foo, user_ref)
    handler = common_ldap._get_connection(CONF.ldap.url, use_pool=True)
    self.assertIsInstance(handler, common_ldap.PooledLDAPHandler)