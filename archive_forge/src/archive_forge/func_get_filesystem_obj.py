from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def get_filesystem_obj(self, nas_server=None, name=None, id=None):
    filesystem = id if id else name
    try:
        obj_fs = None
        if name:
            if not nas_server:
                err_msg = 'NAS Server is required to get the FileSystem.'
                LOG.error(err_msg)
                self.module.fail_json(msg=err_msg)
            obj_fs = self.unity_conn.get_filesystem(name=name, nas_server=nas_server)
            if obj_fs and obj_fs.existed:
                LOG.info('Successfully got the filesystem object %s.', obj_fs)
                return obj_fs
        if id:
            if nas_server:
                obj_fs = self.unity_conn.get_filesystem(id=id, nas_server=nas_server)
            else:
                obj_fs = self.unity_conn.get_filesystem(id=id)
            if obj_fs and obj_fs.existed:
                LOG.info('Successfully got the filesystem object %s.', obj_fs)
                return obj_fs
    except Exception as e:
        error_msg = 'Failed to get filesystem %s with error %s.' % (filesystem, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)