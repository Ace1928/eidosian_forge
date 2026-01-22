from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('roles')
def generate_roles_dict(module, fusion):
    roles_info = {}
    api_instance = purefusion.RolesApi(fusion)
    roles = api_instance.list_roles()
    for role in roles:
        name = role.name
        roles_info[name] = {'display_name': role.display_name, 'scopes': role.assignable_scopes}
    return roles_info