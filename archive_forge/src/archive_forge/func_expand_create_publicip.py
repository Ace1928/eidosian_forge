from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_publicip(d, array_index):
    r = dict()
    v = navigate_value(d, ['eip_id'], array_index)
    if not is_empty_value(v):
        r['id'] = v
    return r