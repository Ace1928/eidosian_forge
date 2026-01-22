from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_extend_disk_os_extend(d, array_index):
    r = dict()
    v = navigate_value(d, ['size'], array_index)
    if not is_empty_value(v):
        r['new_size'] = v
    return r