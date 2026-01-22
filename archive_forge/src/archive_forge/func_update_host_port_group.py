from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def update_host_port_group(self, host_system, portgroup_object):
    """Update a Port Group on a given host
        Args:
            host_system: Name of Host System
        """
    changed = changed_security = False
    changed_list = []
    host_results = dict(changed=False, msg='')
    spec = portgroup_object.spec
    message = ''
    host_results['vlan_id'] = self.vlan_id
    if spec.vlanId != self.vlan_id:
        changed = True
        changed_list.append('VLAN ID')
        host_results['vlan_id_previous'] = spec.vlanId
        spec.vlanId = self.vlan_id
    if self.sec_promiscuous_mode is None:
        host_results['sec_promiscuous_mode'] = 'No override'
    else:
        host_results['sec_promiscuous_mode'] = self.sec_promiscuous_mode
    if self.sec_mac_changes is None:
        host_results['sec_mac_changes'] = 'No override'
    else:
        host_results['sec_mac_changes'] = self.sec_mac_changes
    if self.sec_forged_transmits is None:
        host_results['sec_forged_transmits'] = 'No override'
    else:
        host_results['sec_forged_transmits'] = self.sec_forged_transmits
    if spec.policy.security:
        promiscuous_mode_previous = spec.policy.security.allowPromiscuous
        mac_changes_previous = spec.policy.security.macChanges
        forged_transmits_previous = spec.policy.security.forgedTransmits
        if promiscuous_mode_previous is not self.sec_promiscuous_mode:
            spec.policy.security.allowPromiscuous = self.sec_promiscuous_mode
            changed = changed_security = True
            changed_list.append('Promiscuous mode')
        if mac_changes_previous is not self.sec_mac_changes:
            spec.policy.security.macChanges = self.sec_mac_changes
            changed = changed_security = True
            changed_list.append('MAC address changes')
        if forged_transmits_previous is not self.sec_forged_transmits:
            spec.policy.security.forgedTransmits = self.sec_forged_transmits
            changed = changed_security = True
            changed_list.append('Forged transmits')
        if changed_security:
            if self.sec_promiscuous_mode is None:
                host_results['sec_promiscuous_mode_previous'] = 'No override'
            else:
                host_results['sec_promiscuous_mode_previous'] = promiscuous_mode_previous
            if self.sec_mac_changes is None:
                host_results['sec_mac_changes_previous'] = 'No override'
            else:
                host_results['sec_mac_changes'] = mac_changes_previous
            if self.sec_forged_transmits is None:
                host_results['sec_forged_transmits_previous'] = 'No override'
            else:
                host_results['sec_forged_transmits_previous'] = forged_transmits_previous
    else:
        spec.policy.security = self.create_security_policy()
        changed = True
        changed_list.append('Security')
        host_results['sec_promiscuous_mode_previous'] = 'No override'
        host_results['sec_mac_changes_previous'] = 'No override'
        host_results['sec_forged_transmits_previous'] = 'No override'
    if self.ts_enabled is None:
        host_results['traffic_shaping'] = 'No override'
    else:
        host_results['traffic_shaping'] = self.ts_enabled
        if self.ts_enabled:
            ts_average_bandwidth = self.ts_average_bandwidth * 1000
            ts_peak_bandwidth = self.ts_peak_bandwidth * 1000
            ts_burst_size = self.ts_burst_size * 1024
            host_results['traffic_shaping_avg_bandw'] = ts_average_bandwidth
            host_results['traffic_shaping_peak_bandw'] = ts_peak_bandwidth
            host_results['traffic_shaping_burst'] = ts_burst_size
    if spec.policy.shapingPolicy and spec.policy.shapingPolicy.enabled is not None:
        if spec.policy.shapingPolicy.enabled:
            if self.ts_enabled:
                if spec.policy.shapingPolicy.averageBandwidth != ts_average_bandwidth:
                    changed = True
                    changed_list.append('Average bandwidth')
                    host_results['traffic_shaping_avg_bandw_previous'] = spec.policy.shapingPolicy.averageBandwidth
                    spec.policy.shapingPolicy.averageBandwidth = ts_average_bandwidth
                if spec.policy.shapingPolicy.peakBandwidth != ts_peak_bandwidth:
                    changed = True
                    changed_list.append('Peak bandwidth')
                    host_results['traffic_shaping_peak_bandw_previous'] = spec.policy.shapingPolicy.peakBandwidth
                    spec.policy.shapingPolicy.peakBandwidth = ts_peak_bandwidth
                if spec.policy.shapingPolicy.burstSize != ts_burst_size:
                    changed = True
                    changed_list.append('Burst size')
                    host_results['traffic_shaping_burst_previous'] = spec.policy.shapingPolicy.burstSize
                    spec.policy.shapingPolicy.burstSize = ts_burst_size
            elif self.ts_enabled is False:
                changed = True
                changed_list.append('Traffic shaping')
                host_results['traffic_shaping_previous'] = True
                spec.policy.shapingPolicy.enabled = False
            elif self.ts_enabled is None:
                spec.policy.shapingPolicy = None
                changed = True
                changed_list.append('Traffic shaping')
                host_results['traffic_shaping_previous'] = True
        elif self.ts_enabled:
            spec.policy.shapingPolicy = self.create_shaping_policy()
            changed = True
            changed_list.append('Traffic shaping')
            host_results['traffic_shaping_previous'] = False
        elif self.ts_enabled is None:
            spec.policy.shapingPolicy = None
            changed = True
            changed_list.append('Traffic shaping')
            host_results['traffic_shaping_previous'] = True
    elif self.ts_enabled:
        spec.policy.shapingPolicy = self.create_shaping_policy()
        changed = True
        changed_list.append('Traffic shaping')
        host_results['traffic_shaping_previous'] = 'No override'
    elif self.ts_enabled is False:
        changed = True
        changed_list.append('Traffic shaping')
        host_results['traffic_shaping_previous'] = 'No override'
        spec.policy.shapingPolicy.enabled = False
    if spec.policy.nicTeaming:
        if self.teaming_load_balancing is None:
            host_results['load_balancing'] = 'No override'
        else:
            host_results['load_balancing'] = self.teaming_load_balancing
        if spec.policy.nicTeaming.policy:
            if spec.policy.nicTeaming.policy != self.teaming_load_balancing:
                changed = True
                changed_list.append('Load balancing')
                host_results['load_balancing_previous'] = spec.policy.nicTeaming.policy
                spec.policy.nicTeaming.policy = self.teaming_load_balancing
        elif self.teaming_load_balancing:
            changed = True
            changed_list.append('Load balancing')
            host_results['load_balancing_previous'] = 'No override'
            spec.policy.nicTeaming.policy = self.teaming_load_balancing
        if spec.policy.nicTeaming.notifySwitches is None:
            host_results['notify_switches'] = 'No override'
        else:
            host_results['notify_switches'] = self.teaming_notify_switches
        if spec.policy.nicTeaming.notifySwitches is not None:
            if self.teaming_notify_switches is not None:
                if spec.policy.nicTeaming.notifySwitches is not self.teaming_notify_switches:
                    changed = True
                    changed_list.append('Notify switches')
                    host_results['notify_switches_previous'] = spec.policy.nicTeaming.notifySwitches
                    spec.policy.nicTeaming.notifySwitches = self.teaming_notify_switches
            else:
                changed = True
                changed_list.append('Notify switches')
                host_results['notify_switches_previous'] = spec.policy.nicTeaming.notifySwitches
                spec.policy.nicTeaming.notifySwitches = None
        elif self.teaming_notify_switches is not None:
            changed = True
            changed_list.append('Notify switches')
            host_results['notify_switches_previous'] = 'No override'
            spec.policy.nicTeaming.notifySwitches = self.teaming_notify_switches
        if spec.policy.nicTeaming.rollingOrder is None:
            host_results['failback'] = 'No override'
        else:
            host_results['failback'] = self.teaming_failback
        if spec.policy.nicTeaming.rollingOrder is not None:
            if self.teaming_failback is not None:
                if spec.policy.nicTeaming.rollingOrder is self.teaming_failback:
                    changed = True
                    changed_list.append('Failback')
                    host_results['failback_previous'] = not spec.policy.nicTeaming.rollingOrder
                    spec.policy.nicTeaming.rollingOrder = not self.teaming_failback
            else:
                changed = True
                changed_list.append('Failback')
                host_results['failback_previous'] = spec.policy.nicTeaming.rollingOrder
                spec.policy.nicTeaming.rollingOrder = None
        elif self.teaming_failback is not None:
            changed = True
            changed_list.append('Failback')
            host_results['failback_previous'] = 'No override'
            spec.policy.nicTeaming.rollingOrder = not self.teaming_failback
        if self.teaming_failover_order_active is None and self.teaming_failover_order_standby is None:
            host_results['failover_active'] = 'No override'
            host_results['failover_standby'] = 'No override'
        else:
            host_results['failover_active'] = self.teaming_failover_order_active
            host_results['failover_standby'] = self.teaming_failover_order_standby
        if spec.policy.nicTeaming.nicOrder:
            if self.teaming_failover_order_active or self.teaming_failover_order_standby:
                if spec.policy.nicTeaming.nicOrder.activeNic != self.teaming_failover_order_active:
                    changed = True
                    changed_list.append('Failover order active')
                    host_results['failover_active_previous'] = spec.policy.nicTeaming.nicOrder.activeNic
                    spec.policy.nicTeaming.nicOrder.activeNic = self.teaming_failover_order_active
                if spec.policy.nicTeaming.nicOrder.standbyNic != self.teaming_failover_order_standby:
                    changed = True
                    changed_list.append('Failover order standby')
                    host_results['failover_standby_previous'] = spec.policy.nicTeaming.nicOrder.standbyNic
                    spec.policy.nicTeaming.nicOrder.standbyNic = self.teaming_failover_order_standby
            else:
                spec.policy.nicTeaming.nicOrder = None
                changed = True
                changed_list.append('Failover order')
                if hasattr(spec.policy.nicTeaming.nicOrder, 'activeNic'):
                    host_results['failover_active_previous'] = spec.policy.nicTeaming.nicOrder.activeNic
                else:
                    host_results['failover_active_previous'] = []
                if hasattr(spec.policy.nicTeaming.nicOrder, 'standbyNic'):
                    host_results['failover_standby_previous'] = spec.policy.nicTeaming.nicOrder.standbyNic
                else:
                    host_results['failover_standby_previous'] = []
        elif self.teaming_failover_order_active or self.teaming_failover_order_standby:
            changed = True
            changed_list.append('Failover order')
            host_results['failover_active_previous'] = 'No override'
            host_results['failover_standby_previous'] = 'No override'
            spec.policy.nicTeaming.nicOrder = self.create_nic_order_policy()
        if self.teaming_failure_detection is None:
            host_results['failure_detection'] = 'No override'
        else:
            host_results['failure_detection'] = self.teaming_failure_detection
        if spec.policy.nicTeaming.failureCriteria and spec.policy.nicTeaming.failureCriteria.checkBeacon is not None:
            if self.teaming_failure_detection == 'link_status_only':
                if spec.policy.nicTeaming.failureCriteria.checkBeacon is True:
                    changed = True
                    changed_list.append('Network failure detection')
                    host_results['failure_detection_previous'] = 'beacon_probing'
                    spec.policy.nicTeaming.failureCriteria.checkBeacon = False
            elif self.teaming_failure_detection == 'beacon_probing':
                if spec.policy.nicTeaming.failureCriteria.checkBeacon is False:
                    changed = True
                    changed_list.append('Network failure detection')
                    host_results['failure_detection_previous'] = 'link_status_only'
                    spec.policy.nicTeaming.failureCriteria.checkBeacon = True
            elif spec.policy.nicTeaming.failureCriteria.checkBeacon is not None:
                changed = True
                changed_list.append('Network failure detection')
                host_results['failure_detection_previous'] = spec.policy.nicTeaming.failureCriteria.checkBeacon
                spec.policy.nicTeaming.failureCriteria = None
        elif self.teaming_failure_detection:
            spec.policy.nicTeaming.failureCriteria = self.create_nic_failure_policy()
            changed = True
            changed_list.append('Network failure detection')
            host_results['failure_detection_previous'] = 'No override'
    else:
        spec.policy.nicTeaming = self.create_teaming_policy()
        if spec.policy.nicTeaming:
            changed = True
            changed_list.append('Teaming and failover')
            host_results['load_balancing_previous'] = 'No override'
            host_results['notify_switches_previous'] = 'No override'
            host_results['failback_previous'] = 'No override'
            host_results['failover_active_previous'] = 'No override'
            host_results['failover_standby_previous'] = 'No override'
            host_results['failure_detection_previous'] = 'No override'
    if changed:
        if self.module.check_mode:
            changed_suffix = ' would be changed'
        else:
            changed_suffix = ' changed'
        if len(changed_list) > 2:
            message = ', '.join(changed_list[:-1]) + ', and ' + str(changed_list[-1])
        elif len(changed_list) == 2:
            message = ' and '.join(changed_list)
        elif len(changed_list) == 1:
            message = changed_list[0]
        message += changed_suffix
        if not self.module.check_mode:
            try:
                host_system.configManager.networkSystem.UpdatePortGroup(pgName=self.portgroup, portgrp=spec)
            except vim.fault.AlreadyExists as already_exists:
                self.module.fail_json(msg='Failed to update Portgroup as it would conflict with an existing port group: %s' % to_native(already_exists.msg))
            except vim.fault.NotFound as not_found:
                self.module.fail_json(msg='Failed to update Portgroup as vSwitch was not found: %s' % to_native(not_found.msg))
            except vim.fault.HostConfigFault as host_config_fault:
                self.module.fail_json(msg='Failed to update Portgroup due to host system configuration failure : %s' % to_native(host_config_fault.msg))
            except vmodl.fault.InvalidArgument as invalid_argument:
                self.module.fail_json(msg="Failed to update Port Group '%s', this can be due to either of following : 1. VLAN id was not correct as per specifications, 2. Network policy is invalid : %s" % (self.portgroup, to_native(invalid_argument.msg)))
    else:
        message = 'Port Group already configured properly'
    host_results['changed'] = changed
    host_results['msg'] = message
    host_results['portgroup'] = self.portgroup
    host_results['vswitch'] = self.switch
    return (changed, host_results)