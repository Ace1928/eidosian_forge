from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def get_s3user(module, blade):
    """Return Object Store Account or None"""
    full_user = module.params['account'] + '/' + module.params['name']
    s3user = None
    s3users = blade.object_store_users.list_object_store_users()
    for user in range(0, len(s3users.items)):
        if s3users.items[user].name == full_user:
            s3user = s3users.items[user]
    return s3user