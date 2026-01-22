from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def convert_timestamp_to_sec(expiry_time, snap_time):
    """Converts the time difference to seconds"""
    snap_time_str = snap_time.strftime('%m/%d/%Y %H:%M')
    snap_timestamp = datetime.strptime(snap_time_str, '%m/%d/%Y %H:%M')
    expiry_timestamp = datetime.strptime(expiry_time, '%m/%d/%Y %H:%M')
    return int((expiry_timestamp - snap_timestamp).total_seconds())