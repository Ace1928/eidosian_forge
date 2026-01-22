from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def is_dict_element_present(dict, key):
    """
    This function checks whether the key is present in dict.
    :param dict:
    :param key:
    :return:
    """
    for item in dict:
        if item == key:
            return True
    return False