from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def check_rule_cfg_exists(self, cursor):
    query_string = 'SELECT count(*) AS `rule_count`\n               FROM mysql_query_rules'
    cols = 0
    query_data = []
    for col, val in iteritems(self.config_data):
        if val is not None:
            cols += 1
            query_data.append(val)
            if cols == 1:
                query_string += '\n WHERE ' + col + ' = %s'
            else:
                query_string += '\n  AND ' + col + ' = %s'
    if cols > 0:
        cursor.execute(query_string, query_data)
    else:
        cursor.execute(query_string)
    check_count = cursor.fetchone()
    return int(check_count['rule_count'])