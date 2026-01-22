from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_extendparam(d, array_index):
    r = dict()
    r['chargingMode'] = 0
    v = navigate_value(d, ['enterprise_project_id'], array_index)
    if not is_empty_value(v):
        r['enterprise_project_id'] = v
    v = navigate_value(d, ['enable_auto_recovery'], array_index)
    if not is_empty_value(v):
        r['support_auto_recovery'] = v
    return r