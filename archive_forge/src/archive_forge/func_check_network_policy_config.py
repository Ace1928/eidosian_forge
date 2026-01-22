from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_network_policy_config(self):
    changed_promiscuous = changed_forged_transmits = changed_mac_changes = False
    promiscuous_previous = forged_transmits_previous = mac_changes_previous = None
    current_config = self.dvs.config.defaultPortConfig
    policy = vim.dvs.VmwareDistributedVirtualSwitch.MacManagementPolicy()
    if 'promiscuous' in self.network_policy and current_config.macManagementPolicy.allowPromiscuous != self.network_policy['promiscuous']:
        changed_promiscuous = True
        promiscuous_previous = current_config.macManagementPolicy.allowPromiscuous
        policy.allowPromiscuous = self.network_policy['promiscuous']
    if 'forged_transmits' in self.network_policy and current_config.macManagementPolicy.forgedTransmits != self.network_policy['forged_transmits']:
        changed_forged_transmits = True
        forged_transmits_previous = current_config.macManagementPolicy.forgedTransmits
        policy.forgedTransmits = self.network_policy['forged_transmits']
    if 'mac_changes' in self.network_policy and current_config.macManagementPolicy.macChanges != self.network_policy['mac_changes']:
        changed_mac_changes = True
        mac_changes_previous = current_config.macManagementPolicy.macChanges
        policy.macChanges = self.network_policy['mac_changes']
    changed = changed_promiscuous or changed_forged_transmits or changed_mac_changes
    return (policy, changed, changed_promiscuous, promiscuous_previous, changed_forged_transmits, forged_transmits_previous, changed_mac_changes, mac_changes_previous)