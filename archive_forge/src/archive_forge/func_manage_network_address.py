from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def manage_network_address(self, host_details, network_address_list, network_address, network_address_state):
    try:
        is_mapped = False
        changed = False
        for addr in network_address_list:
            if addr.lower() == network_address.lower():
                is_mapped = True
                break
        if not is_mapped and network_address_state == 'present-in-host':
            LOG.info('Adding network address %s to Host %s', network_address, host_details.name)
            host_details.add_ip_port(network_address)
            changed = True
        elif is_mapped and network_address_state == 'absent-in-host':
            LOG.info('Deleting network address %s from Host %s', network_address, host_details.name)
            host_details.delete_ip_port(network_address)
            changed = True
        if changed:
            updated_host = self.unity.get_host(name=host_details.name)
            network_address_list = self.get_host_network_address_list(updated_host)
        return (network_address_list, changed)
    except Exception as e:
        error_message = 'Got error %s while modifying network address %s of host %s' % (str(e), network_address, host_details.name)
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)