from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def ext_delete(check_mode, cursor, ext, cascade):
    """Remove the extension from the database.

    Return True if success.

    Args:
      cursor (cursor) -- cursor object of psycopg library
      ext (str) -- extension name
      cascade (boolean) -- Pass the CASCADE flag to the DROP command
    """
    query = 'DROP EXTENSION "%s"' % ext
    if cascade:
        query += ' CASCADE'
    if not check_mode:
        cursor.execute(query)
    executed_queries.append(cursor.mogrify(query))
    return True