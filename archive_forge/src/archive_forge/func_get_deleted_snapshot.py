from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from datetime import datetime
def get_deleted_snapshot(module, array, arrayv6):
    """Return Deleted Snapshot"""
    snapname = module.params['name'] + '.' + module.params['suffix']
    if module.params['offload']:
        source_array = list(arrayv6.get_arrays().items)[0].name
        snapname = module.params['name'] + '.' + module.params['suffix']
        full_snapname = source_array + ':' + snapname
        if _check_offload(module, arrayv6):
            res = arrayv6.get_remote_volume_snapshots(on=module.params['offload'], names=[full_snapname], destroyed=True)
        else:
            res = arrayv6.get_volume_snapshots(names=[snapname], destroyed=True)
        if res.status_code == 200:
            return list(res.items)[0].destroyed
        else:
            return False
    else:
        try:
            return bool(array.get_volume(snapname, snap=True, pending=True)[0]['time_remaining'] != '')
        except Exception:
            return False