from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def has_diff_elem2(ls1, ls2):
    for elem in ls2:
        if isinstance(elem, dict):
            find = False
            keys1 = elem.keys()
            for elem2 in ls1:
                keys2 = elem2.keys()
                common_keys = []
                for key in keys1:
                    if key in keys2:
                        common_keys.append(key)
                has_diff = False
                for k in common_keys:
                    if isinstance(elem2[k], str) and have_to_change_to_lowercase(elem2[k].lower()):
                        if elem2[k].lower() != elem[k].lower():
                            has_diff = True
                    elif elem2[k] != elem[k]:
                        has_diff = True
                if not has_diff:
                    find = True
                    break
            if not find:
                return True
    return False