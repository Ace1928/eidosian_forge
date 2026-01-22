from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def get_error_infos(self, cli_errors):
    error_list = []
    for errors in cli_errors.args:
        for error in errors:
            error_code = error[0]
            error_string = error[1]
            error_type = fortios_error_codes.get(error_code, 'unknown')
            error_list.append(dict(error_code=error_code, error_type=error_type, error_string=error_string))
    return error_list