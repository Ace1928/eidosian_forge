from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell import utils
def get_cifs_server_parameters():
    """This method provide parameters required for the ansible
       CIFS server module on Unity"""
    return dict(cifs_server_id=dict(), cifs_server_name=dict(), netbios_name=dict(), workgroup=dict(), local_password=dict(no_log=True), domain=dict(), domain_username=dict(), domain_password=dict(no_log=True), nas_server_name=dict(), nas_server_id=dict(), interfaces=dict(type='list', elements='str'), unjoin_cifs_server_account=dict(type='bool'), state=dict(required=True, type='str', choices=['present', 'absent']))