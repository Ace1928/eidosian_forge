from __future__ import absolute_import, division, print_function
def get_role_type(self, role_type):
    if role_type:
        if role_type == 'custom':
            return 'CustomRole'
        else:
            return 'SystemRole'
    return role_type