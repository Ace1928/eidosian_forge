from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def autostart(self, vmid, as_flag):
    self.conn = self.__get_conn()
    if self.conn.get_autostart(vmid) != as_flag:
        self.conn.set_autostart(vmid, as_flag)
        return True
    return False