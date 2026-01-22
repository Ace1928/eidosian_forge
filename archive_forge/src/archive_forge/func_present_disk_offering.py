from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_disk_offering(self):
    disk_offering = self.get_disk_offering()
    if not disk_offering:
        disk_offering = self._create_offering(disk_offering)
    else:
        disk_offering = self._update_offering(disk_offering)
    return disk_offering