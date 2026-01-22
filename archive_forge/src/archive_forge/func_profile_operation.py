from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def profile_operation(module, rest_obj):
    command = module.params.get('command')
    if command == 'create':
        create_profile(module, rest_obj)
    if command == 'modify':
        modify_profile(module, rest_obj)
    if command == 'delete':
        delete_profile(module, rest_obj)
    if command == 'assign':
        assign_profile(module, rest_obj)
    if command == 'unassign':
        unassign_profile(module, rest_obj)
    if command == 'migrate':
        migrate_profile(module, rest_obj)