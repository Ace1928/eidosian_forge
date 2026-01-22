from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
def is_cidr_notation(self):
    return self.address.count('/') == 1