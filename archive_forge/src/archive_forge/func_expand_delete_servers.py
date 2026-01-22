from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_delete_servers(d, array_index):
    new_ai = dict()
    if array_index:
        new_ai.update(array_index)
    req = []
    n = 1
    for i in range(n):
        transformed = dict()
        v = expand_delete_servers_id(d, new_ai)
        if not is_empty_value(v):
            transformed['id'] = v
        if transformed:
            req.append(transformed)
    return req