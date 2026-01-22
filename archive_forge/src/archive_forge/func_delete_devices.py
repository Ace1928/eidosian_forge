from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict, job_tracking
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import CHANGES_MSG, NO_CHANGES_MSG
def delete_devices(module, rest_obj, valid_ids):
    if module.check_mode:
        module.exit_json(msg=CHANGES_MSG, changed=True)
    payload = {'DeviceIds': list(valid_ids)}
    rest_obj.invoke_request('POST', DELETE_DEVICES_URI, data=payload)
    module.exit_json(msg=DELETE_SUCCESS, changed=True)