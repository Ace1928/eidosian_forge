from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def vnetpeering_to_dict(vnetpeering):
    """
    Convert a virtual network peering object to a dict.
    """
    results = dict(id=vnetpeering.id, name=vnetpeering.name, remote_virtual_network=vnetpeering.remote_virtual_network.id, remote_address_space=dict(address_prefixes=vnetpeering.remote_address_space.address_prefixes), peering_state=vnetpeering.peering_state, provisioning_state=vnetpeering.provisioning_state, use_remote_gateways=vnetpeering.use_remote_gateways, allow_gateway_transit=vnetpeering.allow_gateway_transit, allow_forwarded_traffic=vnetpeering.allow_forwarded_traffic, allow_virtual_network_access=vnetpeering.allow_virtual_network_access, peering_sync_level=vnetpeering.peering_sync_level)
    return results