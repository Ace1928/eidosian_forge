from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def get_key_match_op(key, test_keys):
    k_match_op = __KEY_MATCH_OP_DEFAULT
    t_key_set = set()
    if not test_keys or not key:
        return k_match_op
    t_keys = next((t_key_item[key] for t_key_item in test_keys if key in t_key_item), None)
    if t_keys:
        k_match_op = t_keys.get('__key_match_op', __KEY_MATCH_OP_DEFAULT)
    return k_match_op