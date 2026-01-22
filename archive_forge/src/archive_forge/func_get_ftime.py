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
def get_ftime(module, inp_schedule, time_type, time_interval):
    def_time = '00:00'
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    hhmm = inp_schedule.get(f'time_{time_type}') if time_interval else def_time
    date_x = inp_schedule.get(f'date_{time_type}')
    time_x = None
    if date_x:
        dtime = f'{date_x} {hhmm}:00.000'
        time_x = validate_time(module, dtime, time_format, time_type)
    elif time_interval:
        dtime = f'{hhmm}:00.000'
    else:
        dtime = ''
    return (dtime, time_x)