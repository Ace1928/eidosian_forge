from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.snapshots import (
def create_pp(module, fusion):
    """Create Protection Policy"""
    pp_api_instance = purefusion.ProtectionPoliciesApi(fusion)
    local_rpo = parse_minutes(module, module.params['local_rpo'])
    local_retention = parse_minutes(module, module.params['local_retention'])
    if local_retention < 10:
        module.fail_json(msg='Local Retention must be a minimum of 10 minutes')
    if local_rpo < 10:
        module.fail_json(msg='Local RPO must be a minimum of 10 minutes')
    changed = True
    id = None
    if not module.check_mode:
        if not module.params['display_name']:
            display_name = module.params['name']
        else:
            display_name = module.params['display_name']
        op = pp_api_instance.create_protection_policy(purefusion.ProtectionPolicyPost(name=module.params['name'], display_name=display_name, objectives=[purefusion.RPO(type='RPO', rpo='PT' + str(local_rpo) + 'M'), purefusion.Retention(type='Retention', after='PT' + str(local_retention) + 'M')]))
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    module.exit_json(changed=changed, id=id)