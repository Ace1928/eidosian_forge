from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def get_pending_vgroup(module, array):
    """Get Deleted Volume Group"""
    vgroup = None
    for vgrp in array.list_vgroups(pending=True):
        if vgrp['name'] == module.params['name'] and vgrp['time_remaining']:
            vgroup = vgrp
            break
    return vgroup