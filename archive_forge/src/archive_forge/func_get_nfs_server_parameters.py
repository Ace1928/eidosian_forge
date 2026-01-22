from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_server_parameters():
    """This method provide parameters required for the ansible
       NFS server module on Unity"""
    return dict(nfs_server_id=dict(type='str'), host_name=dict(type='str'), nfs_v4_enabled=dict(type='bool'), is_secure_enabled=dict(type='bool'), kerberos_domain_controller_type=dict(type='str', choices=['UNIX', 'WINDOWS', 'CUSTOM']), kerberos_domain_controller_username=dict(type='str'), kerberos_domain_controller_password=dict(type='str', no_log=True), nas_server_name=dict(type='str'), nas_server_id=dict(type='str'), is_extended_credentials_enabled=dict(type='bool'), remove_spn_from_kerberos=dict(default=True, type='bool'), state=dict(required=True, type='str', choices=['present', 'absent']))