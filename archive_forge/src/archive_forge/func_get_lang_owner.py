from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def get_lang_owner(cursor, lang):
    """Get language owner.

    Args:
        cursor (cursor): psycopg cursor object.
        lang (str): language name.
    """
    query = 'SELECT r.rolname FROM pg_language l JOIN pg_roles r ON l.lanowner = r.oid WHERE l.lanname = %(lang)s'
    cursor.execute(query, {'lang': lang})
    return cursor.fetchone()['rolname']