from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def compare_payload(json_payload, idrac_attr):
    """
    :param json_payload: json payload created for update operation
    :param idrac_attr: idrac user attributes
    case1: always skip password for difference
    case2: as idrac_attr returns privilege in the format of string so
    convert payload to string only for comparision
    :return: bool
    """
    copy_json = json_payload.copy()
    for key, val in dict(copy_json).items():
        split_key = key.split('#')[1]
        if split_key == 'Password':
            is_change_required = True
            break
        if split_key == 'Privilege':
            copy_json[key] = str(val)
    else:
        is_change_required = bool(list(set(copy_json.items()) - set(idrac_attr.items())))
    return is_change_required