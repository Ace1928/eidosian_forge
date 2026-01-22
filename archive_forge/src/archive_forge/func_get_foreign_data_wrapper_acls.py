from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_foreign_data_wrapper_acls(self, fdws):
    query = 'SELECT fdwacl::text FROM pg_catalog.pg_foreign_data_wrapper\n                   WHERE fdwname = ANY (%s) ORDER BY fdwname'
    self.execute(query, (fdws,))
    return [t['fdwacl'] for t in self.cursor.fetchall()]