from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def filter_invalid_names(module, executable, name):
    valid_names = []
    names = name
    if not isinstance(name, list):
        names = [name]
    for name in names:
        if image_exists(module, executable, name):
            valid_names.append(name)
    return valid_names