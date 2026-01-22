from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def is_snap_has_share(self, fs_snap):
    try:
        obj = self.unity_conn.get_nfs_share(snap=fs_snap) or self.unity_conn.get_cifs_share(snap=fs_snap)
        if len(obj) > 0:
            LOG.info('Snapshot has %s nfs/smb share/s', len(obj))
            return True
    except Exception as e:
        msg = 'Failed to get nfs/smb share from filesystem snapshot. error: %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    return False