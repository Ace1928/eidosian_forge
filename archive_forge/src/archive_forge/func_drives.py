from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@property
def drives(self):
    """Retrieve list of drives found in storage pool."""
    drives = None
    try:
        rc, drives = self.request('storage-systems/%s/drives' % self.ssid)
    except Exception as error:
        self.module.fail_json(msg='Failed to fetch disk drives. Array id [%s].  Error[%s].' % (self.ssid, to_native(error)))
    return drives