from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def text_to_dict(text_lines):
    config_values = {}
    for value in text_lines:
        k, v = value.split('\n', 1)
        if k in config_values:
            config_values[k].append(v)
        else:
            config_values[k] = [v]
    return config_values