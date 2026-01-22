from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def delete_ldap_user_group(module):
    """ Delete a ldap user group """
    changed = False
    ldap_group_name = module.params['user_ldap_group_name']
    ldap_group_id = find_user_ldap_group_id(module)
    if not ldap_group_id:
        changed = False
        return changed
    path = f'users/{ldap_group_id}?approved=yes'
    system = get_system(module)
    try:
        system.api.delete(path=path)
        changed = True
    except APICommandFailed as err:
        if err.status_code in [404]:
            changed = False
        else:
            msg = f'An error occurred deleting user_ldap_group_name {ldap_group_name}: {err}'
            module.fail_json(msg)
    return changed