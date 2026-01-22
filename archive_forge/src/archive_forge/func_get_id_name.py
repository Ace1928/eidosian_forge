from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell import utils
def get_id_name(cifs_server_id=None, cifs_server_name=None, netbios_name=None, nas_server_id=None):
    """Get the id_or_name.
        :param: cifs_server_id: The ID of CIFS server
        :param: cifs_server_name: The name of CIFS server
        :param: netbios_name: Name of the SMB server in windows network
        :param: nas_server_id: The ID of NAS server
        :return: Return id_or_name
    """
    if cifs_server_id:
        id_or_name = cifs_server_id
    elif cifs_server_name:
        id_or_name = cifs_server_name
    elif netbios_name:
        id_or_name = netbios_name
    else:
        id_or_name = nas_server_id
    return id_or_name