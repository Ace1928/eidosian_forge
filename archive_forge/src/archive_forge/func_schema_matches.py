from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def schema_matches(cursor, schema, owner, comment):
    if not schema_exists(cursor, schema):
        return False
    else:
        schema_info = get_schema_info(cursor, schema)
        if owner and owner != schema_info['owner']:
            return False
        if comment is not None:
            current_comment = schema_info['comment'] if schema_info['comment'] is not None else ''
            if comment != current_comment:
                return False
        return True