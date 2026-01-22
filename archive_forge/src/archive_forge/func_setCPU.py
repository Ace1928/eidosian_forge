from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setCPU(self, name, cpu):
    self.__get_conn()
    return self.conn.set_CPU(name, cpu)