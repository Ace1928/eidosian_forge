from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native, to_bytes
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def check_if_roles_changed(uinfo, roles, db_name):

    def make_sure_roles_are_a_list_of_dict(roles, db_name):
        output = list()
        for role in roles:
            if isinstance(role, (binary_type, text_type)):
                new_role = {'role': role, 'db': db_name}
                output.append(new_role)
            else:
                output.append(role)
        return output
    roles_as_list_of_dict = make_sure_roles_are_a_list_of_dict(roles, db_name)
    uinfo_roles = uinfo.get('roles', [])
    if sorted(roles_as_list_of_dict, key=lambda roles: sorted(roles.items())) == sorted(uinfo_roles, key=lambda roles: sorted(roles.items())):
        return False
    return True