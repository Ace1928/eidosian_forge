from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def is_same_subnet(self, object_remote, object_present):
    if isinstance(object_remote, list) and len(object_remote) != 2:
        return False
    tokens = object_present.split('/')
    if len(tokens) != 2:
        return False
    try:
        subnet_number = int(tokens[1])
        if subnet_number < 0 or subnet_number > 32:
            return False
        remote_subnet_number = sum((bin(int(x)).count('1') for x in object_remote[1].split('.')))
        if object_remote[0] == tokens[0] and remote_subnet_number == subnet_number:
            return True
    except Exception as e:
        return False
    return False