from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def create_az(module, fusion):
    """Create Availability Zone"""
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    changed = True
    id = None
    if not module.check_mode:
        if not module.params['display_name']:
            display_name = module.params['name']
        else:
            display_name = module.params['display_name']
        azone = purefusion.AvailabilityZonePost(name=module.params['name'], display_name=display_name)
        op = az_api_instance.create_availability_zone(azone, region_name=module.params['region'])
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    module.exit_json(changed=changed, id=id)