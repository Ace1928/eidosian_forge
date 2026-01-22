from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def undefine(self, vmid, flag):
    """ Stop a domain, and then wipe it from the face of the earth.  (delete disk/config file) """
    self.__get_conn()
    return self.conn.undefine(vmid, flag)