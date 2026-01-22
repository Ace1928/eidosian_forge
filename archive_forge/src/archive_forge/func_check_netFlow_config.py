from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_netFlow_config(self):
    """Check NetFlow config"""
    changed = changed_collectorIpAddress = changed_collectorPort = changed_observationDomainId = changed_activeFlowTimeout = changed_idleFlowTimeout = changed_samplingRate = changed_internalFlowsOnly = False
    collectorIpAddress_previous = collectorPort_previous = observationDomainId_previous = activeFlowTimeout_previous = idleFlowTimeout_previous = samplingRate_previous = internalFlowsOnly_previous = None
    current_config = self.dvs.config.ipfixConfig
    if current_config is None:
        new_config = vim.dvs.VmwareDistributedVirtualSwitch.IpfixConfig()
    else:
        new_config = current_config
    if self.netFlow_collector_ip is not None:
        if current_config.collectorIpAddress != self.netFlow_collector_ip:
            changed = changed_collectorIpAddress = True
            collectorIpAddress_previous = current_config.collectorIpAddress
            new_config.collectorIpAddress = self.netFlow_collector_ip
        if current_config.collectorPort != self.netFlow_collector_port:
            changed = changed_collectorPort = True
            collectorPort_previous = current_config.collectorPort
            new_config.collectorPort = self.netFlow_collector_port
        if current_config.observationDomainId != self.netFlow_observation_domain_id:
            changed = changed_observationDomainId = True
            observationDomainId_previous = current_config.observationDomainId
            new_config.observationDomainId = self.netFlow_observation_domain_id
        if current_config.activeFlowTimeout != self.netFlow_active_flow_timeout:
            changed = changed_activeFlowTimeout = True
            activeFlowTimeout_previous = current_config.activeFlowTimeout
            new_config.activeFlowTimeout = self.netFlow_active_flow_timeout
        if current_config.idleFlowTimeout != self.netFlow_idle_flow_timeout:
            changed = changed_idleFlowTimeout = True
            idleFlowTimeout_previous = current_config.idleFlowTimeout
            new_config.idleFlowTimeout = self.netFlow_idle_flow_timeout
        if current_config.samplingRate != self.netFlow_sampling_rate:
            changed = changed_samplingRate = True
            samplingRate_previous = current_config.samplingRate
            new_config.samplingRate = self.netFlow_sampling_rate
        if current_config.internalFlowsOnly != self.netFlow_internal_flows_only:
            changed = changed_internalFlowsOnly = True
            internalFlowsOnly_previous = current_config.internalFlowsOnly
            new_config.internalFlowsOnly = self.netFlow_internal_flows_only
    return (new_config, changed, changed_collectorIpAddress, collectorIpAddress_previous, changed_collectorPort, collectorPort_previous, changed_observationDomainId, observationDomainId_previous, changed_activeFlowTimeout, activeFlowTimeout_previous, changed_idleFlowTimeout, idleFlowTimeout_previous, changed_samplingRate, samplingRate_previous, changed_internalFlowsOnly, internalFlowsOnly_previous)