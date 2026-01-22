from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_read_resp_root_volume(value):
    if not value:
        return None
    result = dict()
    result['device'] = value.get('device')
    result['id'] = value.get('id')
    return result