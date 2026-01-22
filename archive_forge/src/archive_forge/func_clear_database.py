import subprocess
import ldap.modlist
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.identity.backends import ldap as identity_ldap
from keystone.tests import unit
from keystone.tests.unit import test_backend_ldap
def clear_database(self):
    devnull = open('/dev/null', 'w')
    subprocess.call(['ldapdelete', '-x', '-D', CONF.ldap.user, '-H', CONF.ldap.url, '-w', CONF.ldap.password, '-r', CONF.ldap.suffix], stderr=devnull)
    if CONF.ldap.suffix.startswith('ou='):
        tree_dn_attrs = {'objectclass': 'organizationalUnit', 'ou': 'openstack'}
    else:
        tree_dn_attrs = {'objectclass': ['dcObject', 'organizationalUnit'], 'dc': 'openstack', 'ou': 'openstack'}
    create_object(CONF.ldap.suffix, tree_dn_attrs)
    create_object(CONF.ldap.user_tree_dn, {'objectclass': 'organizationalUnit', 'ou': 'Users'})
    create_object(CONF.ldap.group_tree_dn, {'objectclass': 'organizationalUnit', 'ou': 'UserGroups'})