from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def set_loadbalancer_element_state(self, enabled, nsp_name='internallbvm'):
    loadbalancer = self.get_loadbalancer_element(nsp_name=nsp_name)
    if loadbalancer['enabled'] == enabled:
        return loadbalancer
    args = {'id': loadbalancer['id'], 'enabled': enabled}
    if not self.module.check_mode:
        res = self.query_api('configureInternalLoadBalancerElement', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            loadbalancer = self.poll_job(res, 'internalloadbalancerelement')
    self.result['changed'] = True
    return loadbalancer