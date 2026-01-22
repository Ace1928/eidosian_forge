from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def disable_timeout(module, blade):
    """Disable idle timeout"""
    changed = True
    if not module.check_mode:
        res = blade.patch_arrays(flashblade.Array(idle_timeout=0))
        if res.status_code != 200:
            module.fail_json(msg='Failed to disable GUI idle timeout')
    module.exit_json(changed=changed)