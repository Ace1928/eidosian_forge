from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('network_interfaces')
def generate_nics_dict(module, fusion):
    nics_info = {}
    nic_api_instance = purefusion.NetworkInterfacesApi(fusion)
    arrays_api_instance = purefusion.ArraysApi(fusion)
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    regions_api_instance = purefusion.RegionsApi(fusion)
    regions = regions_api_instance.list_regions()
    for region in regions.items:
        azs = az_api_instance.list_availability_zones(region_name=region.name)
        for az in azs.items:
            array_details = arrays_api_instance.list_arrays(availability_zone_name=az.name, region_name=region.name)
            for array_detail in array_details.items:
                array_name = az.name + '/' + array_detail.name
                nics_info[array_name] = {}
                nics = nic_api_instance.list_network_interfaces(availability_zone_name=az.name, region_name=region.name, array_name=array_detail.name)
                for nic in nics.items:
                    nics_info[array_name][nic.name] = {'enabled': nic.enabled, 'display_name': nic.display_name, 'interface_type': nic.interface_type, 'services': nic.services, 'max_speed': nic.max_speed, 'vlan': nic.eth.vlan, 'address': nic.eth.address, 'mac_address': nic.eth.mac_address, 'gateway': nic.eth.gateway, 'mtu': nic.eth.mtu, 'network_interface_group': nic.network_interface_group.name, 'availability_zone': nic.availability_zone.name}
    return nics_info