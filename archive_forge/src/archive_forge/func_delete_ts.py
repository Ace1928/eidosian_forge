from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def delete_ts(module, fusion):
    """Delete Tenant Space"""
    changed = True
    ts_api_instance = purefusion.TenantSpacesApi(fusion)
    if not module.check_mode:
        op = ts_api_instance.delete_tenant_space(tenant_name=module.params['tenant'], tenant_space_name=module.params['name'])
        await_operation(fusion, op)
    module.exit_json(changed=changed)