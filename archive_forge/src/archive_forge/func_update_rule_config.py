from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def update_rule_config(self, cursor):
    query_string = 'UPDATE mysql_query_rules'
    cols = 0
    query_data = []
    for col, val in iteritems(self.config_data):
        if val is not None and col != 'rule_id':
            cols += 1
            query_data.append(val)
            if cols == 1:
                query_string += '\nSET ' + col + '= %s,'
            else:
                query_string += '\n    ' + col + ' = %s,'
    query_string = query_string[:-1]
    query_string += '\nWHERE rule_id = %s'
    query_data.append(self.config_data['rule_id'])
    cursor.execute(query_string, query_data)
    return True