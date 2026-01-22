from __future__ import absolute_import, division, print_function
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
import datetime
import os
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def convert_module_param_to_json_bool(module, dict_param_name, param_name):
    body = {}
    if module.params[param_name] is not None:
        if module.params[param_name]:
            body[dict_param_name] = 'true'
        else:
            body[dict_param_name] = 'false'
    return body