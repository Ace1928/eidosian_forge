from __future__ import (absolute_import, division, print_function)
import json
import re
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def group_validation(module, rest_obj):
    group_name, group_device = (module.params.get('device_group_name'), [])
    query_param = {'$filter': "Name eq '{0}'".format(group_name)}
    group_resp = rest_obj.invoke_request('GET', GROUP_URI, query_param=query_param)
    group = group_resp.json_data['value']
    if group:
        group_id = group[0]['Id']
        resp = rest_obj.invoke_request('GET', GROUP_DEVICE_URI.format(group_id))
        device_group_resp = resp.json_data['value']
        if device_group_resp:
            for device in device_group_resp:
                if device['Type'] == 1000:
                    group_device.append(device['Id'])
        else:
            module.fail_json(msg='There are no device(s) present in this group.')
    else:
        module.fail_json(msg="Unable to complete the operation because the entered target device group name '{0}' is invalid.".format(group_name))
    if not group_device:
        module.fail_json(msg="The requested group '{0}' does not contain devices that support export log.".format(group_name))
    return group_device