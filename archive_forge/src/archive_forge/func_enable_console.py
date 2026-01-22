from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def enable_console(module, array):
    """Enable Console Lockout"""
    changed = False
    if array.get_console_lock_status()['console_lock'] != 'enabled':
        changed = True
        if not module.check_mode:
            try:
                array.enable_console_lock()
            except Exception:
                module.fail_json(msg='Enabling Console Lock failed')
    module.exit_json(changed=changed)