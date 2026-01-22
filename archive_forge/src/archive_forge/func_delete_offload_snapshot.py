from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
def delete_offload_snapshot(module, array):
    """Delete Offloaded Protection Group Snapshot"""
    changed = False
    snapname = module.params['name'] + '.' + module.params['suffix']
    if ':' in module.params['name'] and module.params['offload']:
        if _check_offload(module, array):
            res = array.get_remote_protection_group_snapshots(names=[snapname], on=module.params['offload'])
            if res.status_code != 200:
                module.fail_json(msg='Offload snapshot {0} does not exist on {1}'.format(snapname, module.params['offload']))
            rpg_destroyed = list(res.items)[0].destroyed
            if not module.check_mode:
                if not rpg_destroyed:
                    changed = True
                    res = array.patch_remote_protection_group_snapshots(names=[snapname], on=module.params['offload'], remote_protection_group_snapshot=DestroyedPatchPost(destroyed=True))
                    if res.status_code != 200:
                        module.fail_json(msg='Failed to delete offloaded snapshot {0} on target {1}. Error: {2}'.format(snapname, module.params['offload'], res.errors[0].message))
                    if module.params['eradicate']:
                        res = array.delete_remote_protection_group_snapshots(names=[snapname], on=module.params['offload'])
                        if res.status_code != 200:
                            module.fail_json(msg='Failed to eradicate offloaded snapshot {0} on target {1}. Error: {2}'.format(snapname, module.params['offload'], res.errors[0].message))
                elif module.params['eradicate']:
                    changed = True
                    res = array.delete_remote_protection_group_snapshots(names=[snapname], on=module.params['offload'])
                    if res.status_code != 200:
                        module.fail_json(msg='Failed to eradicate offloaded snapshot {0} on target {1}. Error: {2}'.format(snapname, module.params['offload'], res.errors[0].message))
        else:
            module.fail_json(msg='Offload target {0} does not exist or not connected'.format(module.params['offload']))
    else:
        module.fail_json(msg='Protection Group name not in the correct format')
    module.exit_json(changed=changed)