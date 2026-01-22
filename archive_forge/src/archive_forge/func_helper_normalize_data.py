from __future__ import absolute_import, division, print_function
def helper_normalize_data(data, del_keys=None):
    """
    Delete None parameter or specified keys from data.

    Parameters:
        data: dictionary

    Returns:
        data: falsene parameter removed data
        del_keys: deleted keys
    """
    if del_keys is None:
        del_keys = []
    for key, value in data.items():
        if value is None:
            del_keys.append(key)
    for key in del_keys:
        if key in data.keys():
            del data[key]
    return (data, del_keys)