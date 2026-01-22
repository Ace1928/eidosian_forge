from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def delete_az(module, fusion):
    """Delete Availability Zone"""
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    changed = True
    if not module.check_mode:
        op = az_api_instance.delete_availability_zone(region_name=module.params['region'], availability_zone_name=module.params['name'])
        await_operation(fusion, op)
    module.exit_json(changed=changed)