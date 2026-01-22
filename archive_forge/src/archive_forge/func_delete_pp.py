from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.snapshots import (
def delete_pp(module, fusion):
    """Delete Protection Policy"""
    pp_api_instance = purefusion.ProtectionPoliciesApi(fusion)
    changed = True
    if not module.check_mode:
        if module.params['destroy_snapshots_on_delete']:
            protection_policy = get_pp(module, fusion)
            snapshots_api = purefusion.SnapshotsApi(fusion)
            snapshots = snapshots_api.query_snapshots(protection_policy_id=protection_policy.id)
            for snap in snapshots.items:
                delete_snapshot(fusion, snap, snapshots_api)
        op = pp_api_instance.delete_protection_policy(protection_policy_name=module.params['name'])
        await_operation(fusion, op)
    module.exit_json(changed=changed)