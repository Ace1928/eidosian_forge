from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_server_tags(d, array_index):
    v = navigate_value(d, ['read', 'tags'], array_index)
    if not v:
        return None
    r = dict()
    for item in v:
        v1 = item.split('=')
        if v1:
            r[v1[0]] = v1[1]
    return r