from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_attach_nics_nics(d, array_index):
    ev = d.get('nics')
    if not ev:
        return None
    val = ev
    cv = d['current_state'].get('nics')
    if cv:
        m = [item.get('ip_address') for item in cv]
        val = [item for item in ev if item.get('ip_address') not in m]
    r = []
    for item in val:
        transformed = dict()
        v = item.get('ip_address')
        if not is_empty_value(v):
            transformed['ip_address'] = v
        v = item.get('subnet_id')
        if not is_empty_value(v):
            transformed['subnet_id'] = v
        if transformed:
            r.append(transformed)
    return r