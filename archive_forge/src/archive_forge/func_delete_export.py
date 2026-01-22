from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def delete_export(module, array):
    """Delete a file system export"""
    changed = False
    all_policies = []
    directory = module.params['filesystem'] + ':' + module.params['directory']
    if not module.params['nfs_policy'] and (not module.params['smb_policy']):
        module.fail_json(msg='At least one policy must be provided')
    if module.params['nfs_policy']:
        policy_exists = bool(array.get_directory_exports(export_names=[module.params['name']], policy_names=[module.params['nfs_policy']], directory_names=[directory]).status_code == 200)
        if policy_exists:
            all_policies.append(module.params['nfs_policy'])
    if module.params['smb_policy']:
        policy_exists = bool(array.get_directory_exports(export_names=[module.params['name']], policy_names=[module.params['smb_policy']], directory_names=[directory]).status_code == 200)
        if policy_exists:
            all_policies.append(module.params['smb_policy'])
    if all_policies:
        changed = True
        if not module.check_mode:
            res = array.delete_directory_exports(export_names=[module.params['name']], policy_names=all_policies)
            if res.status_code != 200:
                module.fail_json(msg='Failed to delete file system export {0}. {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)