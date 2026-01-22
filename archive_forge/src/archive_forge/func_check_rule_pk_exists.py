from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def check_rule_pk_exists(self, cursor):
    query_string = 'SELECT count(*) AS `rule_count`\n               FROM mysql_query_rules\n               WHERE rule_id = %s'
    query_data = [self.config_data['rule_id']]
    cursor.execute(query_string, query_data)
    check_count = cursor.fetchone()
    return int(check_count['rule_count']) > 0