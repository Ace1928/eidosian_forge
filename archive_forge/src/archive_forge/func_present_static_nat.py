from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_static_nat(self):
    ip_address = self.get_ip_address()
    if not ip_address['isstaticnat']:
        ip_address = self.create_static_nat(ip_address)
    else:
        ip_address = self.update_static_nat(ip_address)
    return ip_address