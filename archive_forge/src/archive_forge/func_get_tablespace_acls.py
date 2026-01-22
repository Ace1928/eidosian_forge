from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_tablespace_acls(self, tablespaces):
    query = 'SELECT spcacl::text FROM pg_catalog.pg_tablespace\n                   WHERE spcname = ANY (%s) ORDER BY spcname'
    self.execute(query, (tablespaces,))
    return [t['spcacl'] for t in self.cursor.fetchall()]