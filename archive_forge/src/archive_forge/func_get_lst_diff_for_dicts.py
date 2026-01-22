from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def get_lst_diff_for_dicts(want, have, lst):
    """
    This function generates a list containing values
    that are only in want and not in list in have dict
    :param want: dict object to want
    :param have: dict object to have
    :param lst: list the diff on
    :return: new list object with values which are only in want.
    """
    if not have:
        diff = want.get(lst) or []
    else:
        want_elements = want.get(lst) or {}
        have_elements = have.get(lst) or {}
        diff = list_diff_want_only(want_elements, have_elements)
    return diff