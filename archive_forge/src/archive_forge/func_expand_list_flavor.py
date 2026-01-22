from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_list_flavor(d, array_index):
    r = dict()
    v = navigate_value(d, ['flavor_name'], array_index)
    r['id'] = v
    for v in r.values():
        if v is not None:
            return r
    return None