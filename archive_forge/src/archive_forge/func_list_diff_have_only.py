from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def list_diff_have_only(want_list, have_list):
    """
    This function generated the list containing values
    that are only in have list.
    :param want_list:
    :param have_list:
    :return: new list with values which are only in have list
    """
    if have_list and (not want_list):
        diff = have_list
    elif not have_list:
        diff = None
    else:
        diff = [i for i in have_list + want_list if i in have_list and i not in want_list]
    return diff