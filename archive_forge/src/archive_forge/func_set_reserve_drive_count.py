from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def set_reserve_drive_count(self, check_mode=False):
    """Set the reserve drive count for raidDiskPool."""
    changed = False
    if self.raid_level == 'raidDiskPool' and self.reserve_drive_count:
        maximum_count = self.get_maximum_reserve_drive_count()
        if self.reserve_drive_count < 0 or self.reserve_drive_count > maximum_count:
            self.module.fail_json(msg='Supplied reserve drive count is invalid or exceeds the maximum allowed. Note that it may be necessary to wait for expansion operations to complete before the adjusting the reserve drive count. Maximum [%s]. Array [%s].' % (maximum_count, self.ssid))
        if self.reserve_drive_count != self.get_reserve_drive_count():
            changed = True
        if not check_mode:
            try:
                rc, resp = self.request('storage-systems/%s/symbol/setDiskPoolReservedDriveCount' % self.ssid, method='POST', data=dict(volumeGroupRef=self.pool_detail['id'], newDriveCount=self.reserve_drive_count))
            except Exception as error:
                self.module.fail_json(msg='Failed to set reserve drive count for disk pool. Disk Pool [%s]. Array [%s].' % (self.pool_detail['id'], self.ssid))
    return changed