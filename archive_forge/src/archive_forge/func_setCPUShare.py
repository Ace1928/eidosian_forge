from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setCPUShare(self, name, cpu_share):
    self.__get_conn()
    return self.conn.set_CPU_share(name, cpu_share)