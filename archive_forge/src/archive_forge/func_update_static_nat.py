from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def update_static_nat(self, ip_address):
    args = {'virtualmachineid': self.get_vm(key='id'), 'ipaddressid': ip_address['id'], 'vmguestip': self.get_vm_guest_ip(), 'networkid': self.get_network(key='id')}
    ip_address['vmguestip'] = ip_address['vmipaddress']
    if self.has_changed(args, ip_address, ['vmguestip', 'virtualmachineid']):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('disableStaticNat', ipaddressid=ip_address['id'])
            self.poll_job(res, 'staticnat')
            self.query_api('enableStaticNat', **args)
            self.ip_address = None
            ip_address = self.get_ip_address()
    return ip_address