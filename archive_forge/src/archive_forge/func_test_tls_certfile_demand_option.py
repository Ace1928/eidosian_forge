import ldap.modlist
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.tests import unit
from keystone.tests.unit import test_ldap_livetest
def test_tls_certfile_demand_option(self):
    self.config_fixture.config(group='ldap', use_tls=True, tls_cacertdir=None, tls_req_cert='demand')
    PROVIDERS.identity_api = identity.backends.ldap.Identity()
    user = unit.create_user(PROVIDERS.identity_api, 'default', name='fake1', password='fakepass1')
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertEqual(user['id'], user_ref['id'])
    user['password'] = 'fakepass2'
    PROVIDERS.identity_api.update_user(user['id'], user)
    PROVIDERS.identity_api.delete_user(user['id'])
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, user['id'])