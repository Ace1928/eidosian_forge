from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_dedicated_bandwidth(d, array_index, current_value, exclude_output):
    v = navigate_value(d, ['read', 'bandwidth_share_type'], array_index)
    if not (v and v == 'PER'):
        return current_value
    result = current_value
    if not result:
        result = dict()
    if not exclude_output:
        v = navigate_value(d, ['read', 'bandwidth_id'], array_index)
        if v is not None:
            result['id'] = v
    v = navigate_value(d, ['read', 'bandwidth_name'], array_index)
    if v is not None:
        result['name'] = v
    v = navigate_value(d, ['read', 'bandwidth_size'], array_index)
    if v is not None:
        result['size'] = v
    return result if result else current_value