from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def get_merge_op(key, test_keys):
    mrg_op = __MERGE_OP_DEFAULT
    if not test_keys:
        return mrg_op
    if not key:
        key = '__default_ops'
    t_keys = next((t_key_item[key] for t_key_item in test_keys if key in t_key_item), None)
    if t_keys:
        mrg_op = t_keys.get('__merge_op', __MERGE_OP_DEFAULT)
    return mrg_op