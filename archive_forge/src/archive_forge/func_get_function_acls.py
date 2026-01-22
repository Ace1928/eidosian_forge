from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def get_function_acls(self, schema, function_signatures):
    funcnames = [f.split('(', 1)[0] for f in function_signatures]
    if schema:
        query = 'SELECT proacl::text\n                       FROM pg_catalog.pg_proc p\n                       JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace\n                       WHERE nspname = %s AND proname = ANY (%s)\n                       ORDER BY proname, proargtypes'
        self.execute(query, (schema, funcnames))
    else:
        query = 'SELECT proacl::text FROM pg_catalog.pg_proc WHERE proname = ANY (%s) ORDER BY proname, proargtypes'
        self.execute(query)
    return [t['proacl'] for t in self.cursor.fetchall()]