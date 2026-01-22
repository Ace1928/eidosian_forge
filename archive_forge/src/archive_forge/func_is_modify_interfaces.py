from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell import utils
def is_modify_interfaces(self, cifs_server_details):
    """Check if modification is required in existing interfaces
            :param: cifs_server_details: CIFS server details
            :return: Flag indicating if modification is required
        """
    existing_interfaces = []
    if cifs_server_details['file_interfaces']['UnityFileInterfaceList']:
        for interface in cifs_server_details['file_interfaces']['UnityFileInterfaceList']:
            existing_interfaces.append(interface['UnityFileInterface']['id'])
    for interface in self.module.params['interfaces']:
        if interface not in existing_interfaces:
            return True
    return False