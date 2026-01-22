from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_quota_config_details(self, obj_fs):
    """
        Get the quota config ID mapped to the filesystem
        :param obj_fs: Filesystem instance
        :return: Quota config object if exists else None
        """
    try:
        all_quota_config = self.unity_conn.get_quota_config(filesystem=obj_fs)
        fs_id = obj_fs.id
        if len(all_quota_config) == 0:
            LOG.error('The quota_config object for new filesystem is not updated yet.')
            return None
        for quota_config in range(len(all_quota_config)):
            if fs_id and all_quota_config[quota_config].filesystem.id == fs_id and (not all_quota_config[quota_config].tree_quota):
                msg = 'Quota config id for filesystem %s is %s' % (fs_id, all_quota_config[quota_config].id)
                LOG.info(msg)
                return all_quota_config[quota_config]
    except Exception as e:
        errormsg = 'Failed to fetch quota config for filesystem {0}  with error {1}'.format(fs_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)