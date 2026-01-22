from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def update_reader_hostgroup(self, cursor):
    query_string = 'UPDATE mysql_replication_hostgroups SET reader_hostgroup = %s WHERE writer_hostgroup = %s'
    cursor.execute(query_string, (self.reader_hostgroup, self.writer_hostgroup))