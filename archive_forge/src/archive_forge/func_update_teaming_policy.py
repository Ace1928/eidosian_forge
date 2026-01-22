from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def update_teaming_policy(self, spec, results):
    """
        Update the teaming policy according to the parameters
        Args:
            spec: The vSwitch spec
            results: The results dict

        Returns: True if changes have been made, else false
        """
    if not self.params['teaming'] or not spec.policy.nicTeaming:
        return False
    teaming_policy = spec.policy.nicTeaming
    changed = False
    teaming_load_balancing = self.params['teaming'].get('load_balancing')
    teaming_failure_detection = self.params['teaming'].get('network_failure_detection')
    teaming_notify_switches = self.params['teaming'].get('notify_switches')
    teaming_failback = self.params['teaming'].get('failback')
    teaming_failover_order_active = self.params['teaming'].get('active_adapters')
    teaming_failover_order_standby = self.params['teaming'].get('standby_adapters')
    if teaming_load_balancing is not None:
        results['load_balancing'] = teaming_load_balancing
        if teaming_policy.policy != teaming_load_balancing:
            results['load_balancing_previous'] = teaming_policy.policy
            teaming_policy.policy = teaming_load_balancing
            changed = True
    if teaming_notify_switches is not None:
        results['notify_switches'] = teaming_notify_switches
        if teaming_policy.notifySwitches is not teaming_notify_switches:
            results['notify_switches_previous'] = teaming_policy.notifySwitches
            teaming_policy.notifySwitches = teaming_notify_switches
            changed = True
    if teaming_failback is not None:
        results['failback'] = teaming_failback
        current_failback = not teaming_policy.rollingOrder
        if current_failback != teaming_failback:
            results['failback_previous'] = current_failback
            teaming_policy.rollingOrder = not teaming_failback
            changed = True
    if teaming_failover_order_active is not None:
        results['failover_active'] = teaming_failover_order_active
        if teaming_policy.nicOrder.activeNic != teaming_failover_order_active:
            results['failover_active_previous'] = teaming_policy.nicOrder.activeNic
            teaming_policy.nicOrder.activeNic = teaming_failover_order_active
            changed = True
    if teaming_failover_order_standby is not None:
        results['failover_standby'] = teaming_failover_order_standby
        if teaming_policy.nicOrder.standbyNic != teaming_failover_order_standby:
            results['failover_standby_previous'] = teaming_policy.nicOrder.standbyNic
            teaming_policy.nicOrder.standbyNic = teaming_failover_order_standby
            changed = True
    if teaming_failure_detection is not None:
        results['failure_detection'] = teaming_failure_detection
        if teaming_failure_detection == 'link_status_only':
            if teaming_policy.failureCriteria.checkBeacon is True:
                results['failure_detection_previous'] = 'beacon_probing'
                teaming_policy.failureCriteria.checkBeacon = False
                changed = True
        elif teaming_failure_detection == 'beacon_probing':
            if teaming_policy.failureCriteria.checkBeacon is False:
                results['failure_detection_previous'] = 'link_status_only'
                teaming_policy.failureCriteria.checkBeacon = True
                changed = True
    return changed