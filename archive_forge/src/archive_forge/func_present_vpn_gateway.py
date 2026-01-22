from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_vpn_gateway(self):
    vpn_gateway = self.get_vpn_gateway()
    if not vpn_gateway:
        self.result['changed'] = True
        args = {'vpcid': self.get_vpc(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
        if not self.module.check_mode:
            res = self.query_api('createVpnGateway', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vpn_gateway = self.poll_job(res, 'vpngateway')
    return vpn_gateway