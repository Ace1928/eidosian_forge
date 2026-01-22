from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_s3user(module, blade, internal=False):
    """Delete Object Store Account"""
    changed = True
    if not module.check_mode:
        user = module.params['account'] + '/' + module.params['name']
        try:
            blade.object_store_users.delete_object_store_users(names=[user])
        except Exception:
            module.fail_json(msg='Object Store Account {0}: Deletion failed'.format(module.params['name']))
    if internal:
        return
    module.exit_json(changed=changed)