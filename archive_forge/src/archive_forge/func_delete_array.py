from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def delete_array(module, fusion):
    """Delete Array - not currently available"""
    array_api_instance = purefusion.ArraysApi(fusion)
    if not module.check_mode:
        res = array_api_instance.delete_array(region_name=module.params['region'], availability_zone_name=module.params['availability_zone'], array_name=module.params['name'])
        await_operation(fusion, res)
    return True