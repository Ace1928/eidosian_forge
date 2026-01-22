from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def get_module_params_subsection(module_params, tms_config, resource_key=None):
    """
    Helper method to get a specific module_params subsection
    """
    mp = {}
    if tms_config == 'TMS_GLOBAL':
        relevant_keys = ['certificate', 'compression', 'source_interface', 'vrf']
        for key in relevant_keys:
            mp[key] = module_params[key]
    if tms_config == 'TMS_DESTGROUP':
        mp['destination_groups'] = []
        for destgrp in module_params['destination_groups']:
            if destgrp['id'] == resource_key:
                mp['destination_groups'].append(destgrp)
    if tms_config == 'TMS_SENSORGROUP':
        mp['sensor_groups'] = []
        for sensor in module_params['sensor_groups']:
            if sensor['id'] == resource_key:
                mp['sensor_groups'].append(sensor)
    if tms_config == 'TMS_SUBSCRIPTION':
        mp['subscriptions'] = []
        for sensor in module_params['subscriptions']:
            if sensor['id'] == resource_key:
                mp['subscriptions'].append(sensor)
    return mp