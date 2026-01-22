from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def massage_install_data(data):
    default_error_msg = 'No install all data found'
    if len(data) == 1:
        result_data = data[0]
    elif len(data) == 2:
        result_data = data[1]
    else:
        result_data = default_error_msg
    if len(data) == 2 and isinstance(data[1], dict):
        if 'clierror' in data[1].keys():
            result_data = data[1]['clierror']
        elif 'code' in data[1].keys() and data[1]['code'] == '500':
            result_data = data[1]['msg']
        else:
            result_data = default_error_msg
    return result_data