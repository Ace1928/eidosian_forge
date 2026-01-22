from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native, to_bytes
from hashlib import sha1
def delete_user_config(self, cursor):
    query_string = 'DELETE FROM mysql_users\n               WHERE username = %s\n                 AND backend = %s\n                 AND frontend = %s'
    query_data = [self.username, self.backend, self.frontend]
    cursor.execute(query_string, query_data)
    return True