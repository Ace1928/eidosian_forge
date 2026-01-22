from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def get_test_key_set(key, test_keys):
    tst_keys = deepcopy(test_keys)
    t_key_set = set()
    if not tst_keys or not key:
        return t_key_set
    t_keys = next((t_key_item[key] for t_key_item in tst_keys if key in t_key_item), None)
    if t_keys:
        t_keys.pop('__merge_op', None)
        t_keys.pop('__delete_op', None)
        t_keys.pop('__key_match_op', None)
        t_key_set = set(t_keys.keys())
    return t_key_set