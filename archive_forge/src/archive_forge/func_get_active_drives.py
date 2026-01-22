from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def get_active_drives(self, drives):
    """
        return a list of active drives
        if drives is specified, only [] or a subset of disks in drives are returned
        else all available drives for this node or cluster are returned
        """
    if drives is None:
        return list(self.active_drives.values())
    return [drive_id for drive_id, status in drives if status in ['active', 'failed']]