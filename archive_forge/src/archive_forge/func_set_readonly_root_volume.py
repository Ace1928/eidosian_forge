from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def set_readonly_root_volume(inputv, curv):
    if not (inputv and isinstance(inputv, dict)):
        return
    if not (curv and isinstance(curv, dict)):
        return
    inputv['device'] = curv.get('device')
    inputv['volume_id'] = curv.get('volume_id')