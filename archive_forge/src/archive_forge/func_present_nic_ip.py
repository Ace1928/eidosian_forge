from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_nic_ip(self):
    nic = self.get_nic()
    if not self.get_secondary_ip():
        self.result['changed'] = True
        args = {'nicid': nic['id'], 'ipaddress': self.vm_guest_ip}
        if not self.module.check_mode:
            res = self.query_api('addIpToNic', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                nic = self.poll_job(res, 'nicsecondaryip')
                self.vm_guest_ip = nic['ipaddress']
    return nic