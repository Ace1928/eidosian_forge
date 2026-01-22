from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def validate_attr_name(attribute, req_data):
    invalid_attr = {}
    data_dict = {attr['Name']: attr['Value'] for attr in attribute if attr['Name'] in req_data.keys()}
    if not len(data_dict) == len(req_data):
        for key in req_data.keys():
            if key not in data_dict:
                act_key = key.replace('#', '.')
                invalid_attr[act_key] = 'Attribute does not exist.'
    return (data_dict, invalid_attr)