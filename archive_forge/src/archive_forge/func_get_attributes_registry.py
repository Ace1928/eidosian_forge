from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def get_attributes_registry(idrac):
    reggy = {}
    try:
        resp = idrac.invoke_request('/redfish/v1/Registries/ManagerAttributeRegistry', 'GET')
        loc_list = resp.json_data.get('Location', [])
        if loc_list:
            reg_json_uri = loc_list[-1].get('Uri')
            reg_resp = idrac.invoke_request(reg_json_uri, 'GET')
            attr_list = reg_resp.json_data.get('RegistryEntries').get('Attributes')
            reggy = dict(((x['AttributeName'], x) for x in attr_list))
    except Exception:
        reggy = {}
    return reggy