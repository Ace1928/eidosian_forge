from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def enable_ra(module, blade):
    """Enable Remote Assist"""
    changed = True
    if not module.check_mode:
        ra_settings = Support(remote_assist_active=True)
        try:
            blade.support.update_support(support=ra_settings)
        except Exception:
            module.fail_json(msg='Enabling Remote Assist failed')
    module.exit_json(changed=changed)