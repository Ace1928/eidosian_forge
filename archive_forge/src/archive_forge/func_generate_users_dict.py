from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('users')
def generate_users_dict(module, fusion):
    users_info = {}
    api_instance = purefusion.IdentityManagerApi(fusion)
    users = api_instance.list_users()
    for user in users:
        users_info[user.name] = {'display_name': user.display_name, 'email': user.email, 'id': user.id}
    return users_info