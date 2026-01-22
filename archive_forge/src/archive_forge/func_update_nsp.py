from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def update_nsp(self, name=None, state=None, service_list=None):
    nsp = self.get_nsp(name)
    if not service_list and nsp['state'] == state:
        return nsp
    args = {'id': nsp['id'], 'servicelist': service_list, 'state': state}
    if not self.module.check_mode:
        res = self.query_api('updateNetworkServiceProvider', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            nsp = self.poll_job(res, 'networkserviceprovider')
    self.result['changed'] = True
    return nsp