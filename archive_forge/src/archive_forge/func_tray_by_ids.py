from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def tray_by_ids(self):
    """Retrieve list of trays found in storage system and return dictionary of trays keyed by ids."""
    tray_by_ids = {}
    try:
        rc, inventory = self.request('storage-systems/%s/hardware-inventory' % self.ssid)
        for tray in inventory['trays']:
            tray_by_ids.update({tray['trayRef']: {'tray_number': tray['trayId'], 'drawer_count': tray['driveLayout']['numRows'] * tray['driveLayout']['numColumns']}})
    except Exception as error:
        self.module.fail_json(msg='Failed to fetch trays. Array id [%s]. Error [%s].' % (self.ssid, to_native(error)))
    return tray_by_ids