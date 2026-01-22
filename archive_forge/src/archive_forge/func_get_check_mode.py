from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def get_check_mode(module, idrac, idrac_json, sys_json, lc_json):
    scp_response = idrac.export_scp(export_format='JSON', export_use='Default', target='iDRAC,System,LifecycleController', job_wait=True)
    comp = scp_response.json_data['SystemConfiguration']['Components']
    exist_idrac, exist_sys, exist_lc, invalid = ({}, {}, {}, {})
    for cmp in comp:
        if idrac_json and cmp.get('FQDD') == MANAGER_ID:
            exist_idrac, invalid_attr = validate_attr_name(cmp['Attributes'], idrac_json)
            if invalid_attr:
                invalid.update(invalid_attr)
        if sys_json and cmp.get('FQDD') == SYSTEM_ID:
            exist_sys, invalid_attr = validate_attr_name(cmp['Attributes'], sys_json)
            if invalid_attr:
                invalid.update(invalid_attr)
        if lc_json and cmp.get('FQDD') == LC_ID:
            exist_lc, invalid_attr = validate_attr_name(cmp['Attributes'], lc_json)
            if invalid_attr:
                invalid.update(invalid_attr)
    if invalid:
        module.fail_json(msg='Attributes have invalid values.', invalid_attributes=invalid)
    diff_change = [bool(set(exist_idrac.items()) ^ set(idrac_json.items())) or bool(set(exist_sys.items()) ^ set(sys_json.items())) or bool(set(exist_lc.items()) ^ set(lc_json.items()))]
    if module.check_mode and any(diff_change) is True:
        module.exit_json(msg=CHANGES_MSG, changed=True)
    elif module.check_mode and all(diff_change) is False or (not module.check_mode and all(diff_change) is False):
        module.exit_json(msg=NO_CHANGES_MSG)