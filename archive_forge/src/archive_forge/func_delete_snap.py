from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def delete_snap(module, array):
    """Delete a filesystem snapshot"""
    changed = True
    if not module.check_mode:
        snapname = module.params['filesystem'] + ':' + module.params['name'] + '.' + module.params['client'] + '.' + module.params['suffix']
        directory_snapshot = DirectorySnapshotPatch(destroyed=True)
        res = array.patch_directory_snapshots(names=[snapname], directory_snapshot=directory_snapshot)
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete filesystem snapshot {0}. Error: {1}'.format(snapname, res.errors[0].message))
        if module.params['eradicate']:
            eradicate_snap(module, array)
    module.exit_json(changed=changed)