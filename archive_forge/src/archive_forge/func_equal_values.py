from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def equal_values(v1, v2):
    """
    Checks whether types and content of two values are the same. In case of complex objects, the method might be
    called recursively.

    :param v1: first value
    :param v2: second value
    :return: True if types and content of passed values are equal. Otherwise, returns False.
    :rtype: bool
    """
    if is_string(v1) and is_string(v2):
        return to_text(v1) == to_text(v2)
    if type(v1) is not type(v2):
        return False
    value_type = type(v1)
    if value_type == list:
        return equal_lists(v1, v2)
    elif value_type == dict:
        return equal_dicts(v1, v2)
    else:
        return v1 == v2