from __future__ import (absolute_import, division, print_function)
def prepare_dict(obj):
    """
    Removes any keys from a dictionary that are only specific to our use in the module. FortiManager will reject
    requests with these empty/None keys in it.

    :param obj: Dictionary object to be processed.
    :type obj: dict

    :return: Processed dictionary.
    :rtype: dict
    """
    list_of_elems = ['mode', 'adom', 'host', 'username', 'password']
    if isinstance(obj, dict):
        obj = dict(((key, prepare_dict(value)) for key, value in obj.items() if key not in list_of_elems))
    return obj