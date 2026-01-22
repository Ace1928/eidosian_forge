from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def create_server_config(self, cursor):
    query_string = 'INSERT INTO mysql_servers (\n               hostgroup_id,\n               hostname,\n               port'
    cols = 3
    query_data = [self.hostgroup_id, self.hostname, self.port]
    for col, val in iteritems(self.config_data):
        if val is not None:
            cols += 1
            query_data.append(val)
            query_string += ',\n' + col
    query_string += ')\n' + 'VALUES (' + '%s ,' * cols
    query_string = query_string[:-2]
    query_string += ')'
    cursor.execute(query_string, query_data)
    return True