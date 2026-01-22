from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_host_network_address_list(self, host_details):
    network_address_list = []
    if host_details and host_details.host_ip_ports is not None:
        for port in host_details.host_ip_ports:
            network_address_list.append(port.address)
    return network_address_list