from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
def user_to_principal(fusion, user_id):
    """Given a human-readable Fusion user, such as a Pure 1 App ID
    return the associated principal
    """
    id_api_instance = purefusion.IdentityManagerApi(fusion)
    users = id_api_instance.list_users()
    for user in users:
        if user.name == user_id:
            return user.id
    return None