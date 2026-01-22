from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_DC_byid(self, dc_id):
    return self.conn.datacenters.get(id=dc_id)