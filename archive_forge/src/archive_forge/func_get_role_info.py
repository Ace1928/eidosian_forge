from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_role_info(self):
    """Get information about roles (in PgSQL groups and users are roles)."""
    query = "SELECT r.rolname, r.rolsuper, r.rolcanlogin, r.rolvaliduntil, ARRAY(SELECT b.rolname FROM pg_catalog.pg_auth_members AS m JOIN pg_catalog.pg_roles AS b ON (m.roleid = b.oid) WHERE m.member = r.oid) AS memberof FROM pg_catalog.pg_roles AS r WHERE r.rolname !~ '^pg_'"
    res = self.__exec_sql(query)
    rol_dict = {}
    for i in res:
        rol_dict[i['rolname']] = dict(superuser=i['rolsuper'], canlogin=i['rolcanlogin'], valid_until=i['rolvaliduntil'] if i['rolvaliduntil'] else '', member_of=i['memberof'] if i['memberof'] else [])
    self.pg_info['roles'] = rol_dict