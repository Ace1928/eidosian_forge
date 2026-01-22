from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_parameter_acls(self, parameters):
    if self.pg_version < 150000:
        raise Error('PostgreSQL version must be >= 15 for type=parameter. Exit')
    query = 'SELECT paracl::text FROM pg_catalog.pg_parameter_acl\n                   WHERE parname = ANY (%s) ORDER BY parname'
    self.cursor.execute(query, (parameters,))
    return [t['paracl'] for t in self.cursor.fetchall()]