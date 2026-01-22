from __future__ import absolute_import, division, print_function
import re
from random import randint
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, \
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def get_sharing(self, disk, disk_type, disk_index):
    """
        Get the sharing mode of the virtual disk
        Args:
            disk: Virtual disk data object
            disk_type: Disk type of the virtual disk
            disk_index: Disk unit number at which disk needs to be attached

        Returns:
            sharing_mode: The sharing mode of the virtual disk

        """
    sharing = disk.get('sharing')
    if sharing and disk_type != 'eagerzeroedthick' and (disk_type != 'rdm'):
        self.module.fail_json(msg="Invalid 'sharing' mode specified for disk index [%s]. 'disk_mode' must be 'eagerzeroedthick' or 'rdm' when 'sharing'." % disk_index)
    if sharing:
        sharing_mode = 'sharingMultiWriter'
    else:
        sharing_mode = 'sharingNone'
    return sharing_mode