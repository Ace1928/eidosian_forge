from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_server_instance(self, nfs_server_id):
    """Get NFS server instance.
            :param: nfs_server_id: The ID of the NFS server
            :return: Return NFS server instance if exists
        """
    try:
        nfs_server_obj = self.unity_conn.get_nfs_server(_id=nfs_server_id)
        return nfs_server_obj
    except Exception as e:
        error_msg = 'Failed to get the NFS server %s instance with error %s' % (nfs_server_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)