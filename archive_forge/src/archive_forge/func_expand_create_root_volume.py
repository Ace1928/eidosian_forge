from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_root_volume(d, array_index):
    r = dict()
    v = expand_create_root_volume_extendparam(d, array_index)
    if not is_empty_value(v):
        r['extendparam'] = v
    v = navigate_value(d, ['root_volume', 'size'], array_index)
    if not is_empty_value(v):
        r['size'] = v
    v = navigate_value(d, ['root_volume', 'volume_type'], array_index)
    if not is_empty_value(v):
        r['volumetype'] = v
    return r