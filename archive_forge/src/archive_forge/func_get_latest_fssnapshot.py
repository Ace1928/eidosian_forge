from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def get_latest_fssnapshot(module, blade):
    """Get the name of the latest snpshot or None"""
    try:
        filt = "source='" + module.params['name'] + "'"
        all_snaps = blade.file_system_snapshots.list_file_system_snapshots(filter=filt)
        if not all_snaps.items[0].destroyed:
            return all_snaps.items[0].name
        else:
            module.fail_json(msg='Latest snapshot {0} is destroyed. Eradicate or recover this first.'.format(all_snaps.items[0].name))
    except Exception:
        return None