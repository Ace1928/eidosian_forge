from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def rename_policy(module, array):
    """Rename a file system policy"""
    changed = False
    target_exists = bool(array.get_policies(names=[module.params['rename']]).status_code == 200)
    if target_exists:
        module.fail_json(msg='Rename failed - Target policy {0} already exists'.format(module.params['rename']))
    if not module.check_mode:
        changed = True
        if module.params['policy'] == 'nfs':
            res = array.patch_policies_nfs(names=[module.params['name']], policy=flasharray.PolicyPatch(name=module.params['rename']))
            if res.status_code != 200:
                module.fail_json(msg='Failed to rename NFS policy {0} to {1}'.format(module.params['name'], module.params['rename']))
        elif module.params['policy'] == 'smb':
            res = array.patch_policies_smb(names=[module.params['name']], policy=flasharray.PolicyPatch(name=module.params['rename']))
            if res.status_code != 200:
                module.fail_json(msg='Failed to rename SMB policy {0} to {1}'.format(module.params['name'], module.params['rename']))
        elif module.params['policy'] == 'snapshot':
            res = array.patch_policies_snapshot(names=[module.params['name']], policy=flasharray.PolicyPatch(name=module.params['rename']))
            if res.status_code != 200:
                module.fail_json(msg='Failed to rename snapshot policy {0} to {1}'.format(module.params['name'], module.params['rename']))
        else:
            res = array.patch_policies_quota(names=[module.params['name']], policy=flasharray.PolicyPatch(name=module.params['rename']))
            if res.status_code != 200:
                module.fail_json(msg='Failed to rename quota policy {0} to {1}'.format(module.params['name'], module.params['rename']))
    module.exit_json(changed=changed)