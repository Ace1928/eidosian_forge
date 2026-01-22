from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_smart_fabric_details_via_id(module, rest_obj, fabric_id):
    resp = []
    try:
        fabric_path = "{0}('{1}')".format(FABRIC_URI, fabric_id)
        resp_det = rest_obj.invoke_request('GET', fabric_path)
        resp = [resp_det.json_data]
    except HTTPError:
        module.exit_json(msg=INVALID_FABRIC_ID.format(fabric_id), failed=True)
    return resp