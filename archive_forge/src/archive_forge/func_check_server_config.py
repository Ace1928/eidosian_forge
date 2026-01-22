from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def check_server_config(self, cursor):
    query_string = 'SELECT count(*) AS `host_count`\n               FROM mysql_servers\n               WHERE hostgroup_id = %s\n                 AND hostname = %s\n                 AND port = %s'
    query_data = [self.hostgroup_id, self.hostname, self.port]
    for col, val in iteritems(self.config_data):
        if val is not None:
            query_data.append(val)
            query_string += '\n  AND ' + col + ' = %s'
    cursor.execute(query_string, query_data)
    check_count = cursor.fetchone()
    if isinstance(check_count, tuple):
        return int(check_count[0]) > 0
    return int(check_count['host_count']) > 0