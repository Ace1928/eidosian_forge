from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_filesystem_tree_quota_display_attributes(self, tree_quota_id):
    """Display quota tree attributes
            :param tree_quota_id: Quota tree ID
            :return: Quota tree dict to display
        """
    try:
        tree_quota_obj = self.unity_conn.get_tree_quota(_id=tree_quota_id)
        tree_quota_details = tree_quota_obj._get_properties()
        if tree_quota_obj and tree_quota_obj.existed:
            tree_quota_details['soft_limit'] = utils.convert_size_with_unit(int(tree_quota_details['soft_limit']))
            tree_quota_details['hard_limit'] = utils.convert_size_with_unit(int(tree_quota_details['hard_limit']))
            tree_quota_details['filesystem']['UnityFileSystem']['name'] = tree_quota_obj.filesystem.name
            tree_quota_details['filesystem']['UnityFileSystem'].update({'nas_server': {'name': tree_quota_obj.filesystem.nas_server.name, 'id': tree_quota_obj.filesystem.nas_server.id}})
            return tree_quota_details
    except Exception as e:
        errormsg = 'Failed to display quota tree details {0} with error {1}'.format(tree_quota_obj.id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)