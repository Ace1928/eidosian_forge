from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('regions')
def generate_regions_dict(module, fusion):
    regions_api_instance = purefusion.RegionsApi(fusion)
    return {region.name: {'display_name': region.display_name} for region in regions_api_instance.list_regions().items}