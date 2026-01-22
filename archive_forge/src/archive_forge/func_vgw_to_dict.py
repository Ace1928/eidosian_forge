from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def vgw_to_dict(vgw):
    results = dict(id=vgw.id, name=vgw.name, location=vgw.location, gateway_type=vgw.gateway_type, vpn_type=vgw.vpn_type, vpn_gateway_generation=vgw.vpn_gateway_generation, enable_bgp=vgw.enable_bgp, tags=vgw.tags, provisioning_state=vgw.provisioning_state, sku=dict(name=vgw.sku.name, tier=vgw.sku.tier), bgp_settings=dict(asn=vgw.bgp_settings.asn, bgp_peering_address=vgw.bgp_settings.bgp_peering_address, peer_weight=vgw.bgp_settings.peer_weight) if vgw.bgp_settings else None, etag=vgw.etag)
    return results