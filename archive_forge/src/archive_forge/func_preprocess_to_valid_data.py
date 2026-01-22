from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from urllib.parse import quote
import copy
import traceback
def preprocess_to_valid_data(data):
    if isinstance(data, list):
        return [preprocess_to_valid_data(elem) for elem in data]
    elif isinstance(data, dict):
        return {k.replace('-', '_'): preprocess_to_valid_data(v) for k, v in data.items() if k not in EXCLUDED_LIST}
    return data