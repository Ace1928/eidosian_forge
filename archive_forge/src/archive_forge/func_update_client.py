from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_client(module, blade, client):
    """Update API Client"""
    changed = False
    if client.enabled != module.params['enabled']:
        changed = True
        if not module.check_mode:
            res = blade.patch_api_clients(names=[module.params['name']], api_clients=flashblade.ApiClient(enabled=module.params['enabled']))
            if res.status_code != 200:
                module.fail_json(msg='Failed to update API Client {0}'.format(module.params['name']))
    module.exit_json(changed=changed)