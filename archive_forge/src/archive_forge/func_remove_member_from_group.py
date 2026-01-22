from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
def remove_member_from_group(module, rest_obj, group_id, device_id, current_device_list):
    payload_device_list = [each_id for each_id in device_id if each_id in current_device_list]
    if module.check_mode and payload_device_list:
        module.exit_json(msg='Changes found to be applied.', changed=True, group_id=group_id)
    if not payload_device_list:
        module.exit_json(msg='No changes found to be applied.', group_id=group_id)
    payload = {'GroupId': group_id, 'MemberDeviceIds': payload_device_list}
    response = rest_obj.invoke_request('POST', REMOVE_MEMBER_URI, data=payload)
    return response