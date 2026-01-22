from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_read_resp_extra_dhcp_opts(value):
    if not value:
        return None
    result = []
    for item in value:
        val = dict()
        val['opt_name'] = item.get('opt_name')
        val['opt_value'] = item.get('opt_value')
        result.append(val)
    return result