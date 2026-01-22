from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def ext_get_versions(cursor, ext):
    """
    Get the currently created extension version if it is installed
    in the database, its default version (used to update to 'latest'),
    and versions that are available if it is installed on the system.

    Return tuple (current_version, default_version, [list of available versions]).

    Note: the list of available versions contains only versions
          that higher than the current created version.
          If the extension is not created, this list will contain all
          available versions.

    Args:
      cursor (cursor) -- cursor object of psycopg library
      ext (str) -- extension name
    """
    current_version = None
    default_version = None
    params = {}
    params['ext'] = ext
    query = 'SELECT extversion FROM pg_catalog.pg_extension WHERE extname = %(ext)s'
    cursor.execute(query, params)
    res = cursor.fetchone()
    if res:
        current_version = res['extversion']
    query = 'SELECT default_version FROM pg_catalog.pg_available_extensions WHERE name = %(ext)s'
    cursor.execute(query, params)
    res = cursor.fetchone()
    if res:
        default_version = res['default_version']
    query = 'SELECT version FROM pg_catalog.pg_available_extension_versions WHERE name = %(ext)s'
    cursor.execute(query, params)
    available_versions = set((r['version'] for r in cursor.fetchall()))
    if current_version is None:
        current_version = False
    if default_version is None:
        default_version = False
    return (current_version, default_version, available_versions)