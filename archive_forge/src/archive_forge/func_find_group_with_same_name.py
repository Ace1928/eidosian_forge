from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def find_group_with_same_name(groups, name):
    for group in groups:
        if group['name'] == name:
            return (False, group.get('id'))
    return (True, None)