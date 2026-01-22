from __future__ import (absolute_import, division, print_function)
import json
import socket
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def fetch_device_details(module, rest_obj):
    device_id, tag, final_resp = (module.params.get('device_id'), module.params.get('device_service_tag'), {})
    if device_id is None and tag is None:
        key, value = get_chassis_device(module, rest_obj)
        device_id = value
    else:
        key, value = ('Id', device_id) if device_id is not None else ('DeviceServiceTag', tag)
        param_value = '{0} eq {1}'.format(key, value) if key == 'Id' else "{0} eq '{1}'".format(key, value)
        resp = rest_obj.invoke_request('GET', DEVICE_URI, query_param={'$filter': param_value})
        resp_data = resp.json_data.get('value')
        rename_key = 'id' if key == 'Id' else 'service tag'
        if not resp_data:
            module.fail_json(msg=DEVICE_FAIL_MSG.format(rename_key, value))
        if key == 'DeviceServiceTag' and resp_data[0]['DeviceServiceTag'] == tag:
            device_id = resp_data[0]['Id']
        elif key == 'Id' and resp_data[0]['Id'] == device_id:
            device_id = resp_data[0]['Id']
        else:
            module.fail_json(msg=DEVICE_FAIL_MSG.format(rename_key, value))
    try:
        loc_resp = rest_obj.invoke_request('GET', NETWORK_SERVICE_API.format(device_id))
    except HTTPError as err:
        if err.code == 404:
            module.fail_json(msg=NETWORK_SERVICE_FAIL_MSG)
        err_message = json.load(err)
        error_msg = err_message.get('error', {}).get('@Message.ExtendedInfo')
        if error_msg and error_msg[0].get('MessageId') == 'CGEN1004':
            module.fail_json(msg=NETWORK_SERVICE_FAIL_MSG)
    else:
        loc_resp_data = rest_obj.strip_substr_dict(loc_resp.json_data)
        payload = check_mode_validation(module, loc_resp_data, rest_obj)
        final_resp = rest_obj.invoke_request('PUT', NETWORK_SERVICE_API.format(device_id), data=payload)
    return final_resp