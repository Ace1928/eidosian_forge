from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('availability_zones')
def generate_zones_dict(module, fusion):
    zones_info = {}
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    regions_api_instance = purefusion.RegionsApi(fusion)
    regions = regions_api_instance.list_regions()
    for region in regions.items:
        zones = az_api_instance.list_availability_zones(region_name=region.name)
        for zone in zones.items:
            az_name = zone.name
            zones_info[az_name] = {'display_name': zone.display_name, 'region': zone.region.name}
    return zones_info