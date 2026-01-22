from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def schema_exists(cursor, schema):
    query = 'SELECT schema_name FROM information_schema.schemata WHERE schema_name = %(schema)s'
    cursor.execute(query, {'schema': schema})
    return cursor.rowcount == 1