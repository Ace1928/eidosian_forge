from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def exit_group_operation(module, rest_obj, payload, operation):
    group_resp = rest_obj.invoke_request('POST', OP_URI.format(op=operation), data={'GroupModel': payload})
    cid = int(group_resp.json_data)
    time.sleep(SETTLING_TIME)
    try:
        grp = get_ome_group_by_id(rest_obj, cid)
        group = rest_obj.strip_substr_dict(grp)
    except Exception:
        payload['Id'] = cid
        group = payload
    module.exit_json(changed=True, msg=CREATE_SUCCESS.format(op=operation.lower()), group_status=group)