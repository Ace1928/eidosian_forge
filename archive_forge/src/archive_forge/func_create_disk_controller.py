from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def create_disk_controller(self, ctl_type, ctl_number, bus_sharing='noSharing'):
    disk_ctl = None
    if ctl_type in self.scsi_device_type.keys():
        disk_ctl = self.create_scsi_controller(ctl_type, ctl_number, bus_sharing)
    if ctl_type == 'sata':
        disk_ctl = self.create_sata_controller(ctl_number)
    if ctl_type == 'nvme':
        disk_ctl = self.create_nvme_controller(ctl_number)
    return disk_ctl