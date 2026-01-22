from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def create_array(module, fusion):
    """Create Array"""
    array_api_instance = purefusion.ArraysApi(fusion)
    id = None
    if not module.check_mode:
        if not module.params['display_name']:
            display_name = module.params['name']
        else:
            display_name = module.params['display_name']
        array = purefusion.ArrayPost(hardware_type=module.params['hardware_type'], display_name=display_name, host_name=module.params['host_name'], name=module.params['name'], appliance_id=module.params['appliance_id'], apartment_id=module.params['apartment_id'])
        res = array_api_instance.create_array(array, availability_zone_name=module.params['availability_zone'], region_name=module.params['region'])
        res_op = await_operation(fusion, res)
        id = res_op.result.resource.id
    return (True, id)