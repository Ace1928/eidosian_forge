from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_peering_vpc(d, array_index, current_value, exclude_output):
    result = current_value
    has_init_value = True
    if not result:
        result = dict()
        has_init_value = False
    v = navigate_value(d, ['read', 'accept_vpc_info', 'tenant_id'], array_index)
    result['project_id'] = v
    v = navigate_value(d, ['read', 'accept_vpc_info', 'vpc_id'], array_index)
    result['vpc_id'] = v
    if has_init_value:
        return result
    for v in result.values():
        if v is not None:
            return result
    return current_value