from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_group_memberships(self, groups):
    query = 'SELECT roleid, grantor, member, admin_option\n                   FROM pg_catalog.pg_auth_members am\n                   JOIN pg_catalog.pg_roles r ON r.oid = am.roleid\n                   WHERE r.rolname = ANY(%s)\n                   ORDER BY roleid, grantor, member'
    self.execute(query, (groups,))
    return self.cursor.fetchall()