from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_device_state(module, resp, device_id):
    """Get the current state and device type from response."""
    current_state, device_type, invalid_device = (None, None, True)
    for device in resp['report_list']:
        if device['Id'] == int(device_id):
            current_state = device.get('PowerState', None)
            device_type = device['Type']
            invalid_device = False
            break
    if invalid_device:
        module.fail_json(msg="Unable to complete the operation because the entered target device id '{0}' is invalid.".format(device_id))
    if device_type not in (1000, 2000):
        module.fail_json(msg='Unable to complete the operation because power state supports device type 1000 and 2000.')
    return (current_state, device_type)