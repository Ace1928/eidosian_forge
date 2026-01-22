from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
def update_pgsnapshot(module, array):
    """Update Protection Group Snapshot - basically just rename..."""
    changed = True
    if not module.check_mode:
        current_name = module.params['name'] + '.' + module.params['suffix']
        new_name = module.params['name'] + '.' + module.params['target']
        res = array.patch_protection_group_snapshots(names=[current_name], protection_group_snapshot=ProtectionGroupSnapshotPatch(name=new_name))
        if res.status_code != 200:
            module.fail_json(msg='Failed to rename {0} to {1}. Error: {2}'.format(current_name, new_name, res.errors[0].message))
    module.exit_json(changed=changed)