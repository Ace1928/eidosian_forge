import subprocess
import ldap.modlist
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.identity.backends import ldap as identity_ldap
from keystone.tests import unit
from keystone.tests.unit import test_backend_ldap
def test_ldap_dereferencing(self):
    alt_users_ldif = {'objectclass': ['top', 'organizationalUnit'], 'ou': 'alt_users'}
    alt_fake_user_ldif = {'objectclass': ['person', 'inetOrgPerson'], 'cn': 'alt_fake1', 'sn': 'alt_fake1'}
    aliased_users_ldif = {'objectclass': ['alias', 'extensibleObject'], 'aliasedobjectname': 'ou=alt_users,%s' % CONF.ldap.suffix}
    create_object('ou=alt_users,%s' % CONF.ldap.suffix, alt_users_ldif)
    create_object('%s=alt_fake1,ou=alt_users,%s' % (CONF.ldap.user_id_attribute, CONF.ldap.suffix), alt_fake_user_ldif)
    create_object('ou=alt_users,%s' % CONF.ldap.user_tree_dn, aliased_users_ldif)
    self.config_fixture.config(group='ldap', query_scope='sub', alias_dereferencing='never')
    PROVIDERS.identity_api = identity_ldap.Identity()
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, 'alt_fake1')
    self.config_fixture.config(group='ldap', alias_dereferencing='searching')
    PROVIDERS.identity_api = identity_ldap.Identity()
    user_ref = PROVIDERS.identity_api.get_user('alt_fake1')
    self.assertEqual('alt_fake1', user_ref['id'])
    self.config_fixture.config(group='ldap', alias_dereferencing='always')
    PROVIDERS.identity_api = identity_ldap.Identity()
    user_ref = PROVIDERS.identity_api.get_user('alt_fake1')
    self.assertEqual('alt_fake1', user_ref['id'])