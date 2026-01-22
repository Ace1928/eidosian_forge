from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def disable_console(module, array):
    """Disable Console Lock"""
    changed = False
    if array.get_console_lock_status()['console_lock'] == 'enabled':
        changed = True
        if not module.check_mode:
            try:
                array.disable_console_lock()
            except Exception:
                module.fail_json(msg='Disabling Console Lock failed')
    module.exit_json(changed=changed)