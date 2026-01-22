from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_data_volumes(d, array_index):
    v = navigate_value(d, ['read', 'os-extended-volumes:volumes_attached'], array_index)
    if not v:
        return None
    n = len(v)
    result = []
    new_ai = dict()
    if array_index:
        new_ai.update(array_index)
    for i in range(n):
        new_ai['read.os-extended-volumes:volumes_attached'] = i
        val = dict()
        v = navigate_value(d, ['read', 'os-extended-volumes:volumes_attached', 'device'], new_ai)
        val['device'] = v
        v = navigate_value(d, ['read', 'os-extended-volumes:volumes_attached', 'id'], new_ai)
        val['volume_id'] = v
        for v in val.values():
            if v is not None:
                result.append(val)
                break
    return result if result else None