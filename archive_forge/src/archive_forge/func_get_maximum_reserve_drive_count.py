from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_maximum_reserve_drive_count(self):
    """Retrieve the maximum number of reserve drives for storage pool (Only for raidDiskPool)."""
    if self.raid_level != 'raidDiskPool':
        self.module.fail_json(msg='The storage pool must be a raidDiskPool. Pool [%s]. Array [%s].' % (self.pool_detail['id'], self.ssid))
    drives_ids = list()
    if self.pool_detail:
        drives_ids.extend(self.storage_pool_drives)
        for candidate in self.get_expansion_candidate_drives():
            drives_ids.extend(candidate['drives'])
    else:
        candidate = self.get_candidate_drives()
        drives_ids.extend(candidate['driveRefList']['driveRef'])
    drive_count = len(drives_ids)
    maximum_reserve_drive_count = min(int(drive_count * 0.2 + 1), drive_count - 10)
    if maximum_reserve_drive_count > 10:
        maximum_reserve_drive_count = 10
    return maximum_reserve_drive_count