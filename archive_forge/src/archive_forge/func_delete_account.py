from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_account(module, blade):
    """Delete Active directory Account"""
    changed = True
    if not module.check_mode:
        res = blade.delete_active_directory(names=[module.params['name']], local_only=module.params['local_only'])
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete AD Account {0}'.format(module.params['name']))
    module.exit_json(changed=changed)