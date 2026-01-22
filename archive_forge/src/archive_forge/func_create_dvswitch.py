from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def create_dvswitch(self):
    """Create a DVS"""
    changed = True
    results = dict(changed=changed)
    spec = vim.DistributedVirtualSwitch.CreateSpec()
    spec.configSpec = vim.dvs.VmwareDistributedVirtualSwitch.ConfigSpec()
    results['dvswitch'] = self.switch_name
    spec.configSpec.name = self.switch_name
    results['mtu'] = self.mtu
    spec.configSpec.maxMtu = self.mtu
    results['discovery_protocol'] = self.discovery_protocol
    results['discovery_operation'] = self.discovery_operation
    spec.configSpec.linkDiscoveryProtocolConfig = self.create_ldp_spec()
    results['contact'] = self.contact_name
    results['contact_details'] = self.contact_details
    if self.contact_name or self.contact_details:
        spec.configSpec.contact = self.create_contact_spec()
    results['description'] = self.description
    if self.description:
        spec.description = self.description
    results['uplink_quantity'] = self.uplink_quantity
    spec.configSpec.uplinkPortPolicy = vim.DistributedVirtualSwitch.NameArrayUplinkPortPolicy()
    for count in range(1, self.uplink_quantity + 1):
        spec.configSpec.uplinkPortPolicy.uplinkPortName.append('%s%d' % (self.uplink_prefix, count))
    results['uplinks'] = spec.configSpec.uplinkPortPolicy.uplinkPortName
    results['version'] = self.switch_version
    if self.switch_version:
        spec.productInfo = self.create_product_spec(self.switch_version)
    if self.module.check_mode:
        result = 'DVS would be created'
    else:
        network_folder = self.folder_obj
        task = network_folder.CreateDVS_Task(spec)
        try:
            wait_for_task(task)
        except TaskError as invalid_argument:
            self.module.fail_json(msg='Failed to create DVS : %s' % to_native(invalid_argument))
        self.dvs = find_dvs_by_name(self.content, self.switch_name)
        changed_multicast = changed_network_policy = False
        spec = vim.dvs.VmwareDistributedVirtualSwitch.ConfigSpec()
        spec.configVersion = self.dvs.config.configVersion
        results['multicast_filtering_mode'] = self.multicast_filtering_mode
        multicast_filtering_mode = self.get_api_mc_filtering_mode(self.multicast_filtering_mode)
        if self.dvs.config.multicastFilteringMode != multicast_filtering_mode:
            changed_multicast = True
            spec.multicastFilteringMode = multicast_filtering_mode
        spec.multicastFilteringMode = self.get_api_mc_filtering_mode(self.multicast_filtering_mode)
        network_policy = self.network_policy
        if 'promiscuous' in network_policy or 'forged_transmits' in network_policy or 'mac_changes' in network_policy:
            results['network_policy'] = {}
            if 'promiscuous' in network_policy:
                results['network_policy']['promiscuous'] = network_policy['promiscuous']
            if 'forged_transmits' in network_policy:
                results['network_policy']['forged_transmits'] = network_policy['forged_transmits']
            if 'mac_changes' in network_policy:
                results['network_policy']['mac_changes'] = network_policy['mac_changes']
            result = self.check_network_policy_config()
            changed_network_policy = result[1]
            if changed_network_policy:
                if spec.defaultPortConfig is None:
                    spec.defaultPortConfig = vim.dvs.VmwareDistributedVirtualSwitch.VmwarePortConfigPolicy()
                spec.defaultPortConfig.macManagementPolicy = result[0]
        if self.netFlow_collector_ip is not None:
            results['net_flow_collector_ip'] = self.netFlow_collector_ip
            results['net_flow_collector_port'] = self.netFlow_collector_port
            results['net_flow_observation_domain_id'] = self.netFlow_observation_domain_id
            results['net_flow_active_flow_timeout'] = self.netFlow_active_flow_timeout
            results['net_flow_idle_flow_timeout'] = self.netFlow_idle_flow_timeout
            results['net_flow_sampling_rate'] = self.netFlow_sampling_rate
            results['net_flow_internal_flows_only'] = self.netFlow_internal_flows_only
        result = self.check_netFlow_config()
        changed_netFlow = result[1]
        if changed_netFlow:
            spec.ipfixConfig = result[0]
        if changed_multicast or changed_network_policy or changed_netFlow:
            self.update_dvs_config(self.dvs, spec)
        results['health_check_vlan'] = self.health_check_vlan
        results['health_check_teaming'] = self.health_check_teaming
        results['uuid'] = self.dvs.uuid
        result = self.check_health_check_config(self.dvs.config.healthCheckConfig)
        changed_health_check = result[1]
        if changed_health_check:
            self.update_health_check_config(self.dvs, result[0])
        result = 'DVS created'
    self.module.exit_json(changed=changed, result=to_native(result))