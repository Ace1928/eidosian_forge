from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def enable_ph(module, blade):
    """Enable Phone Hone"""
    changed = True
    if not module.check_mode:
        ph_settings = Support(phonehome_enabled=True)
        try:
            blade.support.update_support(support=ph_settings)
        except Exception:
            module.fail_json(msg='Enabling Phone Home failed')
    module.exit_json(changed=changed)