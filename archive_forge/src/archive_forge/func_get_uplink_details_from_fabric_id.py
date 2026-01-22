from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_uplink_details_from_fabric_id(module, rest_obj, fabric_id):
    resp = []
    try:
        resp_det = rest_obj.invoke_request('GET', ALL_UPLINKS_URI.format(fabric_id))
        resp = resp_det.json_data.get('value')
    except HTTPError:
        module.exit_json(msg=INVALID_FABRIC_ID.format(fabric_id), failed=True)
    return resp