from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import string_types
def list_dict_str(value):
    if isinstance(value, (list, dict, string_types)):
        return value
    raise TypeError