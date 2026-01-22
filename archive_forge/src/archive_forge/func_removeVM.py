from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def removeVM(self, vmname):
    self.__get_conn()
    self.setPower(vmname, 'down', 300)
    return self.conn.remove_VM(vmname)