from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
def get_pgsnapshot(module, array):
    """Return Snapshot (active or deleted) or None"""
    snapname = module.params['name'] + '.' + module.params['suffix']
    res = array.get_protection_group_snapshots(names=[snapname])
    if res.status_code == 200:
        return list(res.items)[0]
    else:
        return None