import ldap
from keystone.common import cache
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
def create_group_container(identity_api):
    group_api = identity_api.driver.group
    conn = group_api.get_connection()
    dn = 'ou=Groups,cn=example,cn=com'
    conn.add_s(dn, [('objectclass', ['organizationalUnit']), ('ou', ['Groups'])])