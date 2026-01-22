from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_network_dict(blade):
    net_info = {}
    ports = blade.network_interfaces.list_network_interfaces()
    for portcnt in range(0, len(ports.items)):
        int_name = ports.items[portcnt].name
        if ports.items[portcnt].enabled:
            net_info[int_name] = {'type': ports.items[portcnt].type, 'mtu': ports.items[portcnt].mtu, 'vlan': ports.items[portcnt].vlan, 'address': ports.items[portcnt].address, 'services': ports.items[portcnt].services, 'gateway': ports.items[portcnt].gateway, 'netmask': ports.items[portcnt].netmask}
    return net_info