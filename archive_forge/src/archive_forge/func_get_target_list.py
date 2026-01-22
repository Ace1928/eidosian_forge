from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_target_list(module, rest_obj):
    target_list = None
    if module.params.get('device_service_tags'):
        target_list = get_dev_ids(module, rest_obj, 'device_service_tags', 'DeviceServiceTag')
    elif module.params.get('device_group_names'):
        target_list = get_group_ids(module, rest_obj)
    elif module.params.get('device_ids'):
        target_list = get_dev_ids(module, rest_obj, 'device_ids', 'Id')
    return target_list