from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def extract_error_msg(self, resp):
    error_info = {}
    if resp.body:
        error = resp.json_data.get('error')
        for each_dict_err in error.get('@Message.ExtendedInfo'):
            key = each_dict_err.get('MessageArgs')[0]
            msg = each_dict_err.get('Message')
            if key not in error_info:
                error_info.update({key: msg})
    return error_info