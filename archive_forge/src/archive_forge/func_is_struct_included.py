from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def is_struct_included(struct1, struct2, exclude=None):
    """
    This function compare if the first parameter structure is included in the second.
    The function use every elements of struct1 and validates they are present in the struct2 structure.
    The two structure does not need to be equals for that function to return true.
    Each elements are compared recursively.
    :param struct1:
        type:
            dict for the initial call, can be dict, list, bool, int or str for recursive calls
        description:
            reference structure
    :param struct2:
        type:
            dict for the initial call, can be dict, list, bool, int or str for recursive calls
        description:
            structure to compare with first parameter.
    :param exclude:
        type:
            list
        description:
            Key to exclude from the comparison.
        default: None
    :return:
        type:
            bool
        description:
            Return True if all element of dict 1 are present in dict 2, return false otherwise.
    """
    if isinstance(struct1, list) and isinstance(struct2, list):
        if not struct1 and (not struct2):
            return True
        for item1 in struct1:
            if isinstance(item1, (list, dict)):
                for item2 in struct2:
                    if is_struct_included(item1, item2, exclude):
                        break
                else:
                    return False
            elif item1 not in struct2:
                return False
        return True
    elif isinstance(struct1, dict) and isinstance(struct2, dict):
        if not struct1 and (not struct2):
            return True
        try:
            for key in struct1:
                if not (exclude and key in exclude):
                    if not is_struct_included(struct1[key], struct2[key], exclude):
                        return False
        except KeyError:
            return False
        return True
    elif isinstance(struct1, bool) and isinstance(struct2, bool):
        return struct1 == struct2
    else:
        return to_text(struct1, 'utf-8') == to_text(struct2, 'utf-8')