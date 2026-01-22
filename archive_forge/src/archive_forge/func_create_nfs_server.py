from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def create_nfs_server(self, nas_server_id, host_name=None, nfs_v4_enabled=None, is_secure_enabled=None, kerberos_domain_controller_type=None, kerberos_domain_controller_username=None, kerberos_domain_controller_password=None, is_extended_credentials_enabled=None):
    """Create NFS server.
            :param: nas_server_id: The ID of NAS server.
            :param: host_name: Name of NFS Server.
            :param: nfs_v4_enabled: Indicates whether the NFSv4 is enabled on the NAS server.
            :param: is_secure_enabled: Indicates whether the secure NFS is enabled.
            :param: kerberos_domain_controller_type: Type of Kerberos Domain Controller used for secure NFS service.
            :param: kerberos_domain_controller_username: Kerberos Domain Controller administrator username.
            :param: kerberos_domain_controller_password: Kerberos Domain Controller administrator password.
            :param: is_extended_credentials_enabled: Indicates whether support for more than 16 unix groups in a Unix credential.
        """
    LOG.info('Creating NFS server')
    try:
        if not self.module.check_mode:
            kdc_enum_type = get_enum_kdctype(kerberos_domain_controller_type)
            if kerberos_domain_controller_type == 'UNIX':
                is_extended_credentials_enabled = None
                is_secure_enabled = None
            utils.UnityNfsServer.create(cli=self.unity_conn._cli, nas_server=nas_server_id, host_name=host_name, nfs_v4_enabled=nfs_v4_enabled, is_secure_enabled=is_secure_enabled, kdc_type=kdc_enum_type, kdc_username=kerberos_domain_controller_username, kdc_password=kerberos_domain_controller_password, is_extended_credentials_enabled=is_extended_credentials_enabled)
        return True
    except Exception as e:
        msg = 'Failed to create NFS server with on NAS Server %s with error: %s' % (nas_server_id, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)