from __future__ import absolute_import, division, print_function
import os
import uuid
from ansible.module_utils.basic import AnsibleModule
def load_key_string(key_str):
    ret_dict = {}
    key_str = key_str.strip()
    ret_dict['key'] = key_str
    cut_key = key_str.split()
    if len(cut_key) in [2, 3]:
        if len(cut_key) == 3:
            ret_dict['label'] = cut_key[2]
    else:
        raise Exception('Public key %s is in wrong format' % key_str)
    return ret_dict