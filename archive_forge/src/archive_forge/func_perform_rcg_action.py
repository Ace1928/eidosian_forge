from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def perform_rcg_action(self, rcg_id, rcg_details):
    """Perform failover, reverse, restore or switchover
            :param rcg_id: Unique identifier of the RCG.
            :param rcg_details: RCG details.
            :return: Boolean indicates if RCG action is successful
        """
    rcg_state = self.module.params['rcg_state']
    force = self.module.params['force']
    if rcg_state == 'failover' and rcg_details['failoverType'] != 'Failover':
        return self.failover(rcg_id)
    if rcg_state == 'switchover' and rcg_details['failoverType'] != 'Switchover':
        return self.switchover(rcg_id, force)
    if rcg_state == 'reverse' and rcg_details['failoverType']:
        return self.reverse(rcg_id)
    if rcg_state == 'restore' and rcg_details['failoverType'] != 'None':
        return self.restore(rcg_id)