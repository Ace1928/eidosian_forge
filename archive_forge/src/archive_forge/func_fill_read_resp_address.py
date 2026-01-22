from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_read_resp_address(value):
    if not value:
        return None
    result = []
    for item in value:
        val = dict()
        val['OS-EXT-IPS:port_id'] = item.get('OS-EXT-IPS:port_id')
        val['OS-EXT-IPS:type'] = item.get('OS-EXT-IPS:type')
        val['addr'] = item.get('addr')
        result.append(val)
    return result