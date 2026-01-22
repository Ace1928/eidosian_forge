from __future__ import absolute_import, division, print_function
import re
from ansible.errors import AnsibleFilterError
def replace_keys_from_dict_n_list(data, target, matching_parameter):
    if isinstance(data, dict):
        for key in target:
            for k in list(data.keys()):
                if matching_parameter == 'regex':
                    if re.match(key.get('before'), k):
                        data[key.get('after')] = data.pop(k)
                elif matching_parameter == 'starts_with':
                    if k.startswith(key.get('before')):
                        data[key.get('after')] = data.pop(k)
                elif matching_parameter == 'ends_with':
                    if k.endswith(key.get('before')):
                        data[key.get('after')] = data.pop(k)
                elif k == key.get('before'):
                    data[key.get('after')] = data.pop(k)
        for k, v in data.items():
            replace_keys_from_dict_n_list(v, target, matching_parameter)
    elif isinstance(data, list):
        for i in data:
            replace_keys_from_dict_n_list(i, target, matching_parameter)
    return data