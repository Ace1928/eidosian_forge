from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_allowed_address_pairs(d, array_index):
    new_array_index = dict()
    if array_index:
        new_array_index.update(array_index)
    req = []
    v = navigate_value(d, ['allowed_address_pairs'], new_array_index)
    if not v:
        return req
    n = len(v)
    for i in range(n):
        new_array_index['allowed_address_pairs'] = i
        transformed = dict()
        v = navigate_value(d, ['allowed_address_pairs', 'ip_address'], new_array_index)
        if not is_empty_value(v):
            transformed['ip_address'] = v
        v = navigate_value(d, ['allowed_address_pairs', 'mac_address'], new_array_index)
        if not is_empty_value(v):
            transformed['mac_address'] = v
        if transformed:
            req.append(transformed)
    return req