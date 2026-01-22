from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_applicable_components(rest_obj, dup_payload, module):
    """Get the target array to be used in spawning jobs for update."""
    target_data = []
    dup_url = 'UpdateService/Actions/UpdateService.GetSingleDupReport'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    dup_resp = rest_obj.invoke_request('POST', dup_url, data=dup_payload, headers=headers, api_timeout=60)
    if dup_resp.status_code == 200:
        dup_data = dup_resp.json_data
        file_token = str(dup_payload['SingleUpdateReportFileToken'])
        for device in dup_data:
            for component in device['DeviceReport']['Components']:
                temp_map = {}
                temp_map['Id'] = device['DeviceId']
                temp_map['Data'] = '{0}={1}'.format(component['ComponentSourceName'], file_token)
                temp_map['TargetType'] = {}
                temp_map['TargetType']['Id'] = int(device['DeviceReport']['DeviceTypeId'])
                temp_map['TargetType']['Name'] = str(device['DeviceReport']['DeviceTypeName'])
                target_data.append(temp_map)
    else:
        module.fail_json(msg=APPLICABLE_DUP)
    return target_data