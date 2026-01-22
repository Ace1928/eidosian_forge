from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def update_nic(self, nic):
    ip_address = self.module.params.get('ip_address')
    if not ip_address:
        return nic
    args = {'nicid': nic['id'], 'ipaddress': ip_address}
    if self.has_changed(args, nic, ['ipaddress']):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateVmNicIp', **args)
            if self.module.params.get('poll_async'):
                vm = self.poll_job(res, 'virtualmachine')
                self.nic = self.get_nic_from_result(result=vm)
    return self.nic