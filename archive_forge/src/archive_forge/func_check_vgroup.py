from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def check_vgroup(module, array):
    """Check is the requested VG to create volume in exists"""
    vg_exists = False
    api_version = array._list_available_rest_versions()
    if VGROUPS_API_VERSION in api_version:
        vg_name = module.params['name'].split('/')[0]
        try:
            vgs = array.list_vgroups()
        except Exception:
            module.fail_json(msg='Failed to get volume groups list. Check array.')
        for vgroup in range(0, len(vgs)):
            if vg_name == vgs[vgroup]['name']:
                vg_exists = True
                break
    else:
        module.fail_json(msg='VG volumes are not supported. Please upgrade your FlashArray.')
    return vg_exists