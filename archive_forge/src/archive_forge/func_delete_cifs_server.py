from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell import utils
def delete_cifs_server(self, cifs_server_id, skip_unjoin=None, domain_username=None, domain_password=None):
    """Delete CIFS server.
            :param: cifs_server_id: The ID of the CIFS server
            :param: skip_unjoin: Flag indicating whether to unjoin SMB server account from AD before deletion
            :param: domain_username: The domain username
            :param: domain_password: The domain password
            :return: Return True if CIFS server is deleted
        """
    LOG.info('Deleting CIFS server')
    try:
        if not self.module.check_mode:
            cifs_obj = self.get_cifs_server_instance(cifs_server_id=cifs_server_id)
            cifs_obj.delete(skip_domain_unjoin=skip_unjoin, username=domain_username, password=domain_password)
        return True
    except Exception as e:
        msg = 'Failed to delete CIFS server: %s with error: %s' % (cifs_server_id, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)