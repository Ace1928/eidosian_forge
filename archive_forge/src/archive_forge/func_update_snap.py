from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def update_snap(module, array, snap_detail):
    """Update a filesystem snapshot retention time"""
    changed = True
    snapname = module.params['filesystem'] + ':' + module.params['name'] + '.' + module.params['client'] + '.' + module.params['suffix']
    if module.params['rename']:
        if not module.params['new_client']:
            new_client = module.params['client']
        else:
            new_client = module.params['new_client']
        if not module.params['new_suffix']:
            new_suffix = module.params['suffix']
        else:
            new_suffix = module.params['new_suffix']
        new_snapname = module.params['filesystem'] + ':' + module.params['name'] + '.' + new_client + '.' + new_suffix
        directory_snapshot = DirectorySnapshotPatch(client_name=new_client, suffix=new_suffix)
        if not module.check_mode:
            res = array.patch_directory_snapshots(names=[snapname], directory_snapshot=directory_snapshot)
            if res.status_code != 200:
                module.fail_json(msg='Failed to rename snapshot {0}. Error: {1}'.format(snapname, res.errors[0].message))
            else:
                snapname = new_snapname
    if not module.params['keep_for'] or module.params['keep_for'] == 0:
        keep_for = 0
    elif 300 <= module.params['keep_for'] <= 31536000:
        keep_for = module.params['keep_for'] * 1000
    else:
        module.fail_json(msg='keep_for not in range of 300 - 31536000')
    if not module.check_mode:
        if snap_detail.destroyed:
            directory_snapshot = DirectorySnapshotPatch(destroyed=False)
            res = array.patch_directory_snapshots(names=[snapname], directory_snapshot=directory_snapshot)
            if res.status_code != 200:
                module.fail_json(msg='Failed to recover snapshot {0}. Error: {1}'.format(snapname, res.errors[0].message))
        directory_snapshot = DirectorySnapshotPatch(keep_for=keep_for)
        if snap_detail.time_remaining == 0 and keep_for != 0:
            res = array.patch_directory_snapshots(names=[snapname], directory_snapshot=directory_snapshot)
            if res.status_code != 200:
                module.fail_json(msg='Failed to retention time for snapshot {0}. Error: {1}'.format(snapname, res.errors[0].message))
        elif snap_detail.time_remaining > 0:
            if module.params['rename'] and module.params['keep_for']:
                res = array.patch_directory_snapshots(names=[snapname], directory_snapshot=directory_snapshot)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to retention time for renamed snapshot {0}. Error: {1}'.format(snapname, res.errors[0].message))
    module.exit_json(changed=changed)