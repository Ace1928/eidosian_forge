from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def create_host_port_group(self, host_system):
    """Create Port Group on a given host
        Args:
            host_system: Name of Host System
        """
    host_results = dict(changed=False, msg='')
    if self.module.check_mode:
        host_results['msg'] = 'Port Group would be added'
    else:
        port_group = vim.host.PortGroup.Config()
        port_group.spec = vim.host.PortGroup.Specification()
        port_group.spec.vswitchName = self.switch
        port_group.spec.name = self.portgroup
        port_group.spec.vlanId = self.vlan_id
        port_group.spec.policy = self.create_network_policy()
        try:
            host_system.configManager.networkSystem.AddPortGroup(portgrp=port_group.spec)
            host_results['changed'] = True
            host_results['msg'] = 'Port Group added'
        except vim.fault.AlreadyExists as already_exists:
            self.module.fail_json(msg='Failed to add Portgroup as it already exists: %s' % to_native(already_exists.msg))
        except vim.fault.NotFound as not_found:
            self.module.fail_json(msg='Failed to add Portgroup as vSwitch was not found: %s' % to_native(not_found.msg))
        except vim.fault.HostConfigFault as host_config_fault:
            self.module.fail_json(msg='Failed to add Portgroup due to host system configuration failure : %s' % to_native(host_config_fault.msg))
        except vmodl.fault.InvalidArgument as invalid_argument:
            self.module.fail_json(msg='Failed to add Portgroup as VLAN id was not correct as per specifications: %s' % to_native(invalid_argument.msg))
    host_results['changed'] = True
    host_results['portgroup'] = self.portgroup
    host_results['vswitch'] = self.switch
    host_results['vlan_id'] = self.vlan_id
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
    host_results['traffic_shaping'] = 'No override' if self.ts_enabled is None else self.ts_enabled
    host_results['load_balancing'] = 'No override' if self.teaming_load_balancing is None else self.teaming_load_balancing
    host_results['notify_switches'] = 'No override' if self.teaming_notify_switches is None else self.teaming_notify_switches
    host_results['failback'] = 'No override' if self.teaming_failback is None else self.teaming_failback
    host_results['failover_active'] = 'No override' if self.teaming_failover_order_active is None else self.teaming_failover_order_active
    host_results['failover_standby'] = 'No override' if self.teaming_failover_order_standby is None else self.teaming_failover_order_standby
    host_results['failure_detection'] = 'No override' if self.teaming_failure_detection is None else self.teaming_failure_detection
    return (True, host_results)