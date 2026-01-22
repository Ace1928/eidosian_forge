from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def delete_fs_snapshot(self, fs_snapshot):
    try:
        if self.is_snap_has_share(fs_snapshot):
            msg = 'Filesystem snapshot cannot be deleted because it has nfs/smb share'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        fs_snapshot.delete()
        return None
    except Exception as e:
        error_msg = 'Failed to delete filesystem snapshot [name: %s, id: %s] with error %s.' % (fs_snapshot.name, fs_snapshot.id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)