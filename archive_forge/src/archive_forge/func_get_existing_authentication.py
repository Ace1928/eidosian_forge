from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def get_existing_authentication(cursor, user, host):
    cursor.execute('SELECT VERSION()')
    srv_type = cursor.fetchone()
    if isinstance(srv_type, dict):
        srv_type = list(srv_type.values())
    if 'mariadb' in srv_type[0].lower():
        cursor.execute('select plugin, auth from (\n            select plugin, password as auth from mysql.user where user=%(user)s\n            and host=%(host)s\n            union select plugin, authentication_string as auth from mysql.user where user=%(user)s\n            and host=%(host)s) x group by plugin, auth limit 2\n        ', {'user': user, 'host': host})
    else:
        cursor.execute('select plugin, authentication_string as auth\n            from mysql.user where user=%(user)s and host=%(host)s\n            group by plugin, authentication_string limit 2', {'user': user, 'host': host})
    rows = cursor.fetchall()
    if isinstance(rows, dict):
        rows = list(rows.values())
    if isinstance(rows[0], tuple):
        return {'plugin': rows[0][0], 'plugin_auth_string': rows[0][1]}
    if isinstance(rows[0], dict):
        return {'plugin': rows[0].get('plugin'), 'plugin_auth_string': rows[0].get('auth')}
    return None