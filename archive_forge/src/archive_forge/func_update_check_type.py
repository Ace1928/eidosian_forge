from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def update_check_type(self, cursor):
    try:
        query_string = 'UPDATE mysql_replication_hostgroups SET check_type = %s WHERE writer_hostgroup = %s'
        cursor.execute(query_string, (self.check_type, self.writer_hostgroup))
    except Exception as e:
        pass