from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('role_assignments')
def generate_ras_dict(module, fusion):
    ras_info = {}
    ras_api_instance = purefusion.RoleAssignmentsApi(fusion)
    role_api_instance = purefusion.RolesApi(fusion)
    roles = role_api_instance.list_roles()
    for role in roles:
        ras = ras_api_instance.list_role_assignments(role_name=role.name)
        for assignment in ras:
            name = assignment.name
            ras_info[name] = {'display_name': assignment.display_name, 'role': assignment.role.name, 'scope': assignment.scope.name}
    return ras_info