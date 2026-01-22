from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_DC(self, dc_name):
    return self.conn.datacenters.get(name=dc_name)