from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def create_filesystem_snapshot(self, snap_name, storage_id, description=None, auto_del=None, expiry_time=None, fs_access_type=None):
    try:
        duration = None
        if expiry_time:
            duration = convert_timestamp_to_sec(expiry_time, self.unity_conn.system_time)
            if duration <= 0:
                self.module.fail_json(msg='expiry_time should be after the current system time.')
        fs_snapshot = self.snap_obj.create(cli=self.unity_conn._cli, storage_resource=storage_id, name=snap_name, description=description, is_auto_delete=auto_del, retention_duration=duration, fs_access_type=fs_access_type)
        return fs_snapshot
    except Exception as e:
        error_msg = 'Failed to create filesystem snapshot %s with error %s' % (snap_name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)