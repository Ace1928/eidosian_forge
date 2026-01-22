from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def get_sc(module, fusion):
    """Return Storage Class or None"""
    sc_api_instance = purefusion.StorageClassesApi(fusion)
    try:
        return sc_api_instance.get_storage_class(storage_class_name=module.params['name'], storage_service_name=module.params['storage_service'])
    except purefusion.rest.ApiException:
        return None