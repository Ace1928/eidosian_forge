from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def to_update_expiry_time(fs_snapshot, expiry_time=None):
    """ Check whether to update expiry_time or not"""
    if not expiry_time:
        return False
    if fs_snapshot.expiration_time is None:
        return True
    if convert_timestamp_to_sec(expiry_time, fs_snapshot.expiration_time) != 0:
        return True
    return False