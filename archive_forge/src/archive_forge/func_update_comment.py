from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def update_comment(self, cursor):
    query_string = 'UPDATE mysql_replication_hostgroups SET comment = %s WHERE writer_hostgroup = %s '
    cursor.execute(query_string, (self.comment, self.writer_hostgroup))