from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def createVM(self, name, cluster, os, actiontype):
    self.__get_conn()
    return self.conn.createVM(name, cluster, os, actiontype)