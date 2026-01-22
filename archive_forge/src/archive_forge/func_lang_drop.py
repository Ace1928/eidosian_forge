from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def lang_drop(cursor, lang, cascade):
    """Drops language for db"""
    cursor.execute('SAVEPOINT ansible_pgsql_lang_drop')
    try:
        if cascade:
            query = 'DROP LANGUAGE "%s" CASCADE' % lang
        else:
            query = 'DROP LANGUAGE "%s"' % lang
        executed_queries.append(query)
        cursor.execute(query)
    except Exception:
        cursor.execute('ROLLBACK TO SAVEPOINT ansible_pgsql_lang_drop')
        cursor.execute('RELEASE SAVEPOINT ansible_pgsql_lang_drop')
        return False
    cursor.execute('RELEASE SAVEPOINT ansible_pgsql_lang_drop')
    return True