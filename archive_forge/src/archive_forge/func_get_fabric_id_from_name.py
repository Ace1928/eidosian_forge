from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_fabric_id_from_name(module, rest_obj, fabric_name):
    fabric_id = ''
    try:
        resp_det = rest_obj.invoke_request('GET', FABRIC_URI)
        resp = resp_det.json_data.get('value')
        for each in resp:
            if each['Name'] == fabric_name:
                fabric_id = each['Id']
                break
    except HTTPError:
        module.exit_json(msg=UNSUCCESS_MSG, failed=True)
    if not fabric_id:
        module.exit_json(msg=INVALID_FABRIC_NAME.format(fabric_name), failed=True)
    return fabric_id