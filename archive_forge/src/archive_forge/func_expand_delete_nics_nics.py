from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_delete_nics_nics(d, array_index):
    cv = d['current_state'].get('nics')
    if not cv:
        return None
    val = cv
    ev = d.get('nics')
    if ev:
        m = [item.get('ip_address') for item in ev]
        val = [item for item in cv if item.get('ip_address') not in m]
    r = []
    for item in val:
        transformed = dict()
        v = item.get('port_id')
        if not is_empty_value(v):
            transformed['id'] = v
        if transformed:
            r.append(transformed)
    return r