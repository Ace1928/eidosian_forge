from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_update_secondary_dns(d, array_index):
    v = navigate_value(d, ['dns_address'], array_index)
    return v[1] if v and len(v) > 1 else ''