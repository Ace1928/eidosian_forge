from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
from ipaddress import ip_network
def get_role_enum(role):
    """Getting correct enum values for role
        :param: role: Indicates role of interface.
        :return: enum value for role.
    """
    if utils.FileInterfaceRoleEnum[role]:
        role = utils.FileInterfaceRoleEnum[role]
        return role