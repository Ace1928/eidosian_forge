from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_specified_device_identifiers(module):
    if module.params.get('device_ids'):
        device_id_list = get_device_id_list(module)
        return {'ids': device_id_list, 'hostnames': []}
    elif module.params.get('hostnames'):
        hostname_list = get_hostname_list(module)
        return {'hostnames': hostname_list, 'ids': []}