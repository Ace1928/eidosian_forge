from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native, to_bytes
from hashlib import sha1
def update_user_config(self, cursor):
    query_string = 'UPDATE mysql_users'
    cols = 0
    query_data = []
    for col, val in iteritems(self.config_data):
        if val is not None:
            cols += 1
            query_data.append(val)
            if cols == 1:
                query_string += '\nSET ' + col + '= %s,'
            else:
                query_string += '\n    ' + col + ' = %s,'
    query_string = query_string[:-1]
    query_string += '\nWHERE username = %s\n  AND backend = %s' + '\n  AND frontend = %s'
    query_data.append(self.username)
    query_data.append(self.backend)
    query_data.append(self.frontend)
    cursor.execute(query_string, query_data)
    return True