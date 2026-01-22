from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@property
def expandable_drive_count(self):
    """Maximum number of drives that a storage pool can be expanded at a given time."""
    capabilities = None
    if self.raid_level == 'raidDiskPool':
        return len(self.available_drives)
    try:
        rc, capabilities = self.request('storage-systems/%s/capabilities' % self.ssid)
    except Exception as error:
        self.module.fail_json(msg='Failed to fetch maximum expandable drive count. Array id [%s].  Error[%s].' % (self.ssid, to_native(error)))
    return capabilities['featureParameters']['maxDCEDrives']