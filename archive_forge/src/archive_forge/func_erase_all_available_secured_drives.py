from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def erase_all_available_secured_drives(self, check_mode=False):
    """Erase all available drives that have encryption at rest feature enabled."""
    changed = False
    drives_list = list()
    for drive in self.drives:
        if drive['available'] and drive['fdeEnabled']:
            changed = True
            drives_list.append(drive['id'])
    if drives_list and (not check_mode):
        try:
            rc, resp = self.request('storage-systems/%s/symbol/reprovisionDrive?verboseErrorResponse=true' % self.ssid, method='POST', data=dict(driveRef=drives_list))
        except Exception as error:
            self.module.fail_json(msg='Failed to erase all secured drives. Array [%s]' % self.ssid)
    return changed