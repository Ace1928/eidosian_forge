from __future__ import absolute_import, division, print_function
import re
from ansible.errors import AnsibleFilterError
def clear_empty_data(data):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = clear_empty_data(v)
    if isinstance(data, list):
        temp = []
        for i in data:
            if i:
                temp.append(clear_empty_data(i))
        return temp
    return data