from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_secondary_ip(self):
    nic = self.get_nic()
    if self.vm_guest_ip:
        secondary_ips = nic.get('secondaryip') or []
        for secondary_ip in secondary_ips:
            if secondary_ip['ipaddress'] == self.vm_guest_ip:
                return secondary_ip
    return None