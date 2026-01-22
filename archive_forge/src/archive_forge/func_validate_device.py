from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def validate_device(module, report, device_id=None, service_tag=None, base_id=None):
    for each in report.get('value'):
        if each['Id'] == device_id:
            break
        if each['ServiceTag'] == service_tag:
            device_id = each['Id']
            break
    else:
        device_name = device_id if device_id is not None else service_tag
        module.fail_json(msg="Unable to complete the operation because the entered target device id or service tag '{0}' is invalid.".format(device_name))
    return device_id