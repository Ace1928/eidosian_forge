from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_filesystem_user_quota_display_attributes(self, user_quota_id):
    """Get display user quota attributes
            :param user_quota_id: User quota ID
            :return: User quota dict to display
        """
    try:
        user_quota_obj = self.unity_conn.get_user_quota(user_quota_id)
        user_quota_details = user_quota_obj._get_properties()
        if user_quota_obj and user_quota_obj.existed:
            user_quota_details['soft_limit'] = utils.convert_size_with_unit(int(user_quota_details['soft_limit']))
            user_quota_details['hard_limit'] = utils.convert_size_with_unit(int(user_quota_details['hard_limit']))
            user_quota_details['filesystem']['UnityFileSystem']['name'] = user_quota_obj.filesystem.name
            user_quota_details['filesystem']['UnityFileSystem'].update({'nas_server': {'name': user_quota_obj.filesystem.nas_server.name, 'id': user_quota_obj.filesystem.nas_server.id}})
            if user_quota_obj.tree_quota:
                user_quota_details['tree_quota']['UnityTreeQuota']['path'] = user_quota_obj.tree_quota.path
            return user_quota_details
        else:
            errormsg = 'User quota does not exist.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
    except Exception as e:
        errormsg = 'Failed to display the details of user quota {0} with error {1}'.format(user_quota_obj.id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)