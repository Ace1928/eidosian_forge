from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def vlan_to_dict(vlan):
    return {'id': vlan.id, 'name': vlan.name, 'description': vlan.description, 'location': vlan.location.id, 'private_ipv4_base_address': vlan.private_ipv4_range_address, 'private_ipv4_prefix_size': vlan.private_ipv4_range_size, 'private_ipv4_gateway_address': vlan.ipv4_gateway, 'ipv6_base_address': vlan.ipv6_range_address, 'ipv6_prefix_size': vlan.ipv6_range_size, 'ipv6_gateway_address': vlan.ipv6_gateway, 'status': vlan.status}