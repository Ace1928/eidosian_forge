from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_list_fixed_ips(d, array_index):
    new_array_index = dict()
    if array_index:
        new_array_index.update(array_index)
    req = []
    n = 1
    for i in range(n):
        transformed = dict()
        v = navigate_value(d, ['ip_address'], new_array_index)
        transformed['ip_address'] = v
        for v in transformed.values():
            if v is not None:
                req.append(transformed)
                break
    return req if req else None