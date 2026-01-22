from __future__ import (absolute_import, division, print_function)
import csv
import os
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination, strip_substr_dict
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
from datetime import datetime
def get_target_payload(module, rest_obj):
    target_payload = {'AllTargets': False, 'DeviceTypes': [], 'Devices': [], 'Groups': [], 'UndiscoveredTargets': []}
    mparams = module.params
    target_provided = False
    if mparams.get('all_devices'):
        target_payload['AllTargets'] = True
        target_provided = True
    elif mparams.get('any_undiscovered_devices'):
        target_payload['UndiscoveredTargets'] = ['ALL_UNDISCOVERED_TARGETS']
        target_provided = True
    elif mparams.get('specific_undiscovered_devices'):
        target_payload['UndiscoveredTargets'] = list(set(module.params.get('specific_undiscovered_devices')))
        target_payload['UndiscoveredTargets'].sort()
        target_provided = True
    elif mparams.get('device_service_tag'):
        devicetype, deviceids = validate_ome_data(module, rest_obj, mparams.get('device_service_tag'), 'DeviceServiceTag', ('Type', 'Id'), DEVICES_URI, 'devices')
        target_payload['Devices'] = deviceids
        target_payload['Devices'].sort()
        target_payload['DeviceTypes'] = list(set(devicetype))
        target_payload['DeviceTypes'].sort()
        target_provided = True
    elif mparams.get('device_group'):
        groups = validate_ome_data(module, rest_obj, mparams.get('device_group'), 'Name', ('Id',), GROUPS_URI, 'groups')
        target_payload['Groups'] = groups[0]
        target_payload['Groups'].sort()
        target_provided = True
    if not target_provided:
        target_payload = {}
    return target_payload