from __future__ import absolute_import, division, print_function
import re
from ansible.errors import AnsibleFilterError
def remove_keys_from_dict_n_list(data, target, matching_parameter):
    if isinstance(data, dict):
        for key in set(target):
            for k in list(data.keys()):
                if matching_parameter == 'regex':
                    if re.match(key, k):
                        del data[k]
                elif matching_parameter == 'starts_with':
                    if k.startswith(key):
                        del data[k]
                elif matching_parameter == 'ends_with':
                    if k.endswith(key):
                        del data[k]
                elif k == key:
                    del data[k]
        for k, v in data.items():
            remove_keys_from_dict_n_list(v, target, matching_parameter)
    elif isinstance(data, list):
        for i in data:
            remove_keys_from_dict_n_list(i, target, matching_parameter)
    return data