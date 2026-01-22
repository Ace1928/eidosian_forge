from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def dnac_compare_equality2(current_value, requested_value, is_query_param=False):
    if is_query_param:
        return True
    if requested_value is None and current_value is None:
        return True
    if requested_value is None:
        return False
    if current_value is None:
        return False
    if isinstance(current_value, dict) and isinstance(requested_value, dict):
        all_dict_params = list(current_value.keys()) + list(requested_value.keys())
        return not any((not fn_comp_key2(param, current_value, requested_value) for param in all_dict_params))
    elif isinstance(current_value, list) and isinstance(requested_value, list):
        return compare_list(current_value, requested_value)
    else:
        return current_value == requested_value