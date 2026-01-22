from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_loadbalancer_element(self, nsp_name='internallbvm'):
    nsp = self.get_nsp(nsp_name)
    nspid = nsp['id']
    if self.loadbalancers is None:
        self.loadbalancers = dict()
        res = self.query_api('listInternalLoadBalancerElements')
        for loadbalancer in res['internalloadbalancerelement']:
            self.loadbalancers[loadbalancer['nspid']] = loadbalancer
        if nspid not in self.loadbalancers:
            self.module.fail_json(msg="Failed: No Loadbalancer found for nsp '%s'" % nsp_name)
    return self.loadbalancers[nspid]