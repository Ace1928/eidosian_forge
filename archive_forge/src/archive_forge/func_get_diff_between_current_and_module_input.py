from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def get_diff_between_current_and_module_input(self, module_attr, server_attr):
    diff, invalid = (0, {})
    if module_attr is None:
        module_attr = {}
    for each_attr in module_attr:
        if each_attr in server_attr:
            data_type = type(server_attr[each_attr])
            if not isinstance(module_attr[each_attr], data_type):
                diff += 1
            elif isinstance(module_attr[each_attr], dict) and isinstance(server_attr[each_attr], dict):
                tmp_diff, tmp_invalid = self.get_diff_between_current_and_module_input(module_attr[each_attr], server_attr[each_attr])
                diff += tmp_diff
                invalid.update(tmp_invalid)
            elif module_attr[each_attr] != server_attr[each_attr]:
                diff += 1
        elif each_attr not in server_attr:
            invalid.update({each_attr: ATTRIBUTE_NOT_EXIST_CHECK_IDEMPOTENCY_MODE})
    return (diff, invalid)