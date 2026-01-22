from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_user_ldap_group(module):
    """
    Find the LDAP user group by name
    """
    result = None
    user_ldap_group_name = module.params['user_ldap_group_name']
    path = f'users?name={user_ldap_group_name}&type=eq%3ALdap'
    system = get_system(module)
    api_result = system.api.get(path=path)
    if len(api_result.get_json()['result']) > 0:
        result = api_result.get_json()['result'][0]
    return result