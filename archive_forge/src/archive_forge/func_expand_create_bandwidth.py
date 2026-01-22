from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_bandwidth(d, array_index):
    v = navigate_value(d, ['dedicated_bandwidth'], array_index)
    sbwid = navigate_value(d, ['shared_bandwidth_id'], array_index)
    if v and sbwid:
        raise Exception("don't input shared_bandwidth_id and dedicated_bandwidth at same time")
    if not (v or sbwid):
        raise Exception('must input shared_bandwidth_id or dedicated_bandwidth')
    if sbwid:
        return {'id': sbwid, 'share_type': 'WHOLE'}
    return {'charge_mode': v['charge_mode'], 'name': v['name'], 'share_type': 'PER', 'size': v['size']}