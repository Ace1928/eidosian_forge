from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def ext_create(check_mode, cursor, ext, schema, cascade, version):
    """
    Create the extension objects inside the database.

    Return True if success.

    Args:
      cursor (cursor) -- cursor object of psycopg library
      ext (str) -- extension name
      schema (str) -- target schema for extension objects
      cascade (boolean) -- Pass the CASCADE flag to the CREATE command
      version (str) -- extension version
    """
    query = 'CREATE EXTENSION "%s"' % ext
    params = {}
    if schema:
        query += ' WITH SCHEMA "%s"' % schema
    if version != 'latest':
        query += ' VERSION %(ver)s'
        params['ver'] = version
    if cascade:
        query += ' CASCADE'
    if not check_mode:
        cursor.execute(query, params)
    executed_queries.append(cursor.mogrify(query, params))
    return True