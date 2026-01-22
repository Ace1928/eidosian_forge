from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def get_valid_groups(module, rest_obj, group_arg, group_set):
    parent = {}
    static_root = {}
    group_dict = {}
    group_resp = rest_obj.get_all_items_with_pagination(GROUP_URI)
    if module.params.get('state') == 'absent':
        group_dict = dict([(str(g[group_arg]).lower(), g) for g in group_resp.get('value') if str(g[group_arg]).lower() in group_set])
    else:
        parg = module.params.get('parent_group_id')
        if parg:
            pkey = 'Id'
        else:
            pkey = 'Name'
            parg = module.params.get('parent_group_name')
        count = 0
        for g in group_resp.get('value'):
            if str(g[group_arg]).lower() in group_set:
                group_dict = g
                count = count + 1
            if str(g[pkey]).lower() == str(parg).lower():
                parent = g
                count = count + 1
            if g['Name'] == STATIC_ROOT:
                static_root = g
                count = count + 1
            if count == 3:
                break
    return (group_dict, parent, static_root)