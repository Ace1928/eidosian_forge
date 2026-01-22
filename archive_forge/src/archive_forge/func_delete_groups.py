from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def delete_groups(rest_obj, module, group_set, group_dict):
    deletables = []
    invalids = []
    for g in group_set:
        grp = group_dict.get(str(g).lower())
        if grp:
            if is_valid_static_group(grp):
                deletables.append(grp['Id'])
            else:
                invalids.append(g)
    if invalids:
        module.fail_json(msg=INVALID_GROUPS_DELETE, invalid_groups=invalids)
    if module.check_mode:
        module.exit_json(changed=True, msg=CHANGES_FOUND, group_ids=deletables)
    rest_obj.invoke_request('POST', OP_URI.format(op='Delete'), data={'GroupIds': deletables})
    module.exit_json(changed=True, msg=DELETE_SUCCESS, group_ids=deletables)