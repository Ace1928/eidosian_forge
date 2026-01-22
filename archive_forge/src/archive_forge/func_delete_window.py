from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_window(module, array):
    """Delete Maintenance Window"""
    changed = False
    if list(array.get_maintenance_windows().items):
        changed = True
        if not module.check_mode:
            state = array.delete_maintenance_windows(names=['environment'])
            if state.status_code != 200:
                changed = False
    module.exit_json(changed=changed)