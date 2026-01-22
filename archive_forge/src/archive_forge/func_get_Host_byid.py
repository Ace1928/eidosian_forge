from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_Host_byid(self, host_id):
    return self.conn.hosts.get(id=host_id)